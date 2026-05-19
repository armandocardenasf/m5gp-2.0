# *********************************************************************
# Name: m5gp.py
# Description: Modulo principal del sistema que implementa los
# metodos del ciclo evolutivo de GP, asi como la interface tipo SkLearn
# Se implementa la logica de ejecucion para funciones de numba y CuML
# *********************************************************************

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os
import sys 
import math
import copy
import pandas as pd
import numpy as np
import time
import gc
try:
  import cupy as cp
  from numba import cuda
  from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float32)
  #import torch
  GPU_CUPY = True
except ImportError:
  GPU_CUPY = False  



# import rmm 
# from rmm.allocators.cupy import rmm_cupy_allocator

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(this_dir))
import m5gpGlobals as gpG
import m5gpCudaMethods as gpCuda
import m5gpCumlMethods as gpCuM
import m5gpCumlMethods2 as gpCuM2
import m5gpMod1 as gpM1
import m5gpMod2 as gpM2
import m5gpMod3 as gpM3


"""
    M5GP Regressor. Modelo de programación genética combinado con un modelo de cuML para regresion simbolica.
"""
class m5gpRegressor(BaseEstimator):
  #method to initialize the class
  def __init__(self, 
            generations=50, 
            Individuals=500, 
            GenesIndividuals=1024, 
            mutationProb=0.15, 
            mutationDeleteRateProb=0.01, 
            sizeTournament=0.20,  
            evaluationMethod=0,        
            scorer=0,  
            maxRandomConstant=1, 
            genOperatorProb=0.54, 
            genVariableProb=0.35, 
            genConstantProb=0.10, 
            genNoopProb=0.001,  
            useOpIF=0,
            functions_set = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "abs", "sum","prod", "avg", "std"],   
            log=1, 
            verbose=1, 
            logPath='log/'
            ):

    env = dict(os.environ)
    self.generations = generations
    self.Individuals=Individuals 
    self.GenesIndividuals=GenesIndividuals
    self.mutationProb=mutationProb 
    self.mutationDeleteRateProb=mutationDeleteRateProb  
    self.evaluationMethod=evaluationMethod
    self.scorer = scorer
    self.sizeTournament=sizeTournament 
    self.maxRandomConstant=maxRandomConstant 
    self.genOperatorProb=genOperatorProb 
    self.genVariableProb=genVariableProb 
    self.genConstantProb=genConstantProb 
    self.genNoopProb=genNoopProb  
    self.useOpIF=useOpIF
    self.nvar=0 
    self.nrowTrain=0 
    self.nrowTest=0
    self.log=log
    self.verbose=verbose
    self.logPath=logPath
    self.functions_set=functions_set
    self.model = ''
    self.m4gpModel = ''
    self.cuModel = 0
    self.bestIndividual = ''

    print("Initializing m5gp")

    # Check if CUDA and Device GPU are available
    if not gpG.cudaSetup(0):
      print("Check CUDA device. Fail to initialize.")
      return

    # Verifica los operadores validos y construye el diccionario a utilizar
    # para generar la poblacion inicial
    if (len(self.functions_set) == 0):
      print("No se definieron operadores")
      exit(0)
    
    # Get valid functions (mathematical operators) allowed for generate individuals
    self.valid_functions_set = gpG.construir_lista_operadores_validos(self.functions_set)
    if (len(self.valid_functions_set) == 0):
      print("No se especificaron operadores validos")
      exit(0)
  

    fName = "M5GP_OpS.csv"
    if os.path.exists(fName):
        os.remove(fName)
    return


  #This method implement the evolution with M5GP  
  def fit(self, X_train, y_train):
    """
    Ejecuta el proceso completo de generación y entrenamiento del modelo M5GP.

    Este método recibe las matrices de datos X y Y que se utilizan tanto para
    la generación de la población inicial como para el entrenamiento evolutivo
    del modelo. Durante el proceso se crea la población, se genera la semántica
    en GPU, se evalúan individuos, se aplican operadores genéticos, se actualizan
    los pesos de operadores (UMAD) y finalmente se ajusta un modelo lineal sobre
    la mejor transformación encontrada.

    Parámetros
    ----------
    X_train : array-like, cuDF.DataFrame o cupy.array
        Matriz con los datos de las características de entrada, usadas para generar 
        la población, evaluar los individuos y construir la matriz semántica durante 
        el proceso evolutivo. Dimensión esperada: (n_muestras, n_caracteristicas).

    y_train : array-like, cuDF.Series o cupy.array
        Vector objetivo asociado a X_train. Se utiliza para calcular la aptitud
        (fitness) de cada individuo y para ajustar el modelo lineal final.
        Dimensión esperada: (n_muestras, ).

    Retorna
    -------
    self : objeto M5GP
        El modelo entrenado, listo para ser utilizado en predicción mediante
        el método `predict()`.

    Notas
    -----
    - La evaluación y construcción de la semántica se ejecuta en GPU mediante
      kernels Numba.
    - El ajuste del modelo lineal se realiza con cuML.
    - Se almacena internamente el mejor individuo y la semántica generada.
    - El modelo resultante es lineal en parámetros, pero basado en características
      simbólicas evolucionadas.
    """

    # Se validan los valores asignados para las probabilidades de generacion de
    # Funciones matematicas, Variables, constantes y operadores NOOP
    # Se normalizan la probabilidades por seguridad
    totalProb = self.genOperatorProb + self.genVariableProb + self.genConstantProb + self.genNoopProb
    if totalProb <= 0.0:
          # fallback razonable
      p_op_n, p_var_n, p_const_n, p_noop_n = 0.53, 0.37, 0.05, 0.05
    else:
      inv = 1.0 / totalProb
      p_op_n, p_var_n, p_const_n, p_noop_n = self.genOperatorProb*inv, self.genVariableProb*inv, self.genConstantProb*inv, self.genNoopProb*inv

    # Se reasignan las probabilidades normalizadas
    self.genOperatorProb=p_op_n 
    self.genVariableProb=p_var_n 
    self.genConstantProb=p_const_n 
    self.genNoopProb=p_noop_n
    #self.maxRandomConstant= np.float32(self.maxRandomConstant)   

    self.X_train = X_train
    self.y_train = y_train

    # train data    
    data=pd.DataFrame(self.X_train)
    data['target']=self.y_train

    self.nrowTrain = len(data.index)
    self.nrowTest = len(data.index)
    self.nrowPredict = len(data.index)
    self.nvar = data.shape[1] - 1
 
    print("Executing Fit - Method(", self.evaluationMethod ,") - ", gpCuM.cuGetMethodName(self), " Scorer:", self.scorer)
    print("nRows:", self.nrowTrain, "nVars:", self.nvar)

    #Initialize operators weigth
    pesos_por_id = {op: 1.0/len(self.valid_functions_set) for op in self.valid_functions_set}
    op_weights = np.array([pesos_por_id[int(oid)] for oid in self.valid_functions_set], dtype=np.float32)
      
    # Prepara la CDF (Cumulative Distribution Function) una vez por generación (Numba)
    # CDF => lista de probabilidades acumuladas que se usa para hacer selección aleatoria ponderada.
    # Esto te permite hacer selección basada en probabilidad directamente.
    cdf = gpM2.preparar_operadores_numba(
        op_ids=self.valid_functions_set, op_weights=op_weights,
        epsilon=0.02, temperatura=1.0
    )
    
    # print("Pesos iniciales")
    # print(pesos_por_id)
    # print(op_weights)
    # print(cdf)

    # Store the size in bytes for initial population
    gpG.sizePopulation = self.Individuals * self.GenesIndividuals 
    gpG.sizeIndividuals = self.Individuals 
    gpG.sizeTournament = math.ceil(self.sizeTournament * self.Individuals)

    # Define vectors to work on device 
    self.model = np.zeros((self.GenesIndividuals ), dtype=np.float32) 

    #print("Initialize Individual")
    # *************************** Initialize population ********************************* 
    hInitialPopulation = gpM1.initialize_population(
                              self.Individuals,
                              self.nvar,
                              self.GenesIndividuals,
                              self.maxRandomConstant,
                              self.genOperatorProb,
                              self.genVariableProb,
                              self.genConstantProb,
                              self.genNoopProb,
                              self.useOpIF,
                              self.valid_functions_set,
                              cdf )
    # -- End of Initialize population --

    # print("Individuals:")
    # print(hInitialPopulation)
    # return
  
    # ***************************  Compute Individuals  ****************************
    hOutIndividuals = [] 
    hStack = []
    hStackIdx = []
    hStackModel = []
  

    #print ("Compute Individual")
    hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
            hInitialPopulation,
            self.X_train,
            self.Individuals,
            self.GenesIndividuals,
            self.nrowTrain,
            self.nvar,
            0 )
    # ****************** End of Compute Individuals **********************
    
    # Get the semantic matrix
    coefArr_p = []
    intercepArr_p = []    
    cuModel_p = []  
    stackBestModel_p = []

    coefArrNew = []
    intercepArrNew = [] 
    cuModelNew = []
    stackBestModelNew = []

    hFit = np.zeros((gpG.sizeIndividuals), dtype=np.float32)
    hFitNew = np.zeros((gpG.sizeIndividuals), dtype=np.float32)
    indexBestOffspring = 0
    indexWorstOffspring = 0


    #print("Compute Error")
    # ***************************** Compute ERROR ***********************************
    hFit, indexBestOffspring, indexWorstOffspring, coefArr_p, intercepArr_p, cuModel_p = gpM1.ComputeError(self,
                hOutIndividuals, 
                self.y_train, 
                self.Individuals, 
                self.nrowTrain,
                hStack, 
                hStackIdx,
                self.evaluationMethod)

    # Index of best individual of initialization
    indexBestIndividual_p = indexBestOffspring 

    del hStack
    del hStackIdx
    gc.collect()

    ajFit = 0
    if (self.evaluationMethod == 1) : #or (self.scorer == 2) :  
      ajFit = gpG.MAX_R2_NEG * (-1)
    trainFit = hFit[indexBestIndividual_p] - ajFit    
    print("Initial Index:", indexBestIndividual_p, " Initial Fit:", trainFit)


    # ***********************************************************************
    # ********************* GP Process Generation Cycle *********************
    # ***********************************************************************
    print ("Starting generational process")
    for generation in range(1,self.generations + 1):
      trainFit = 0
      testFit = 0
      coefArrNew = []
      intercepArrNew = [] 
      cuModelNew = []
      stackBestModelNew = []
      start_time = time.time()

      #print("Torneo")
      # *********************  Select Tournament  **********************
      hNewPopulation, hBestParentsTournament = gpM1.select_tournament(
                    hInitialPopulation,
                    hFit,
                    self.Individuals, 
                    self.GenesIndividuals )

      #print("Mutacion")
      # *********************  UMAD Mutation  **********************
      hNewPopulation = gpM1.umadMutation(self,
                                  hInitialPopulation,
                                  hBestParentsTournament,
                                  self.Individuals,
                                  cdf) 

      #print (hNewPopulation)
      # ***************************  Compute Individuals  ****************************
      hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
              hNewPopulation,
              self.X_train,
              self.Individuals,
              self.GenesIndividuals,
              self.nrowTrain,
              self.nvar,
              0 )
      
      # ***************************** Compute ERROR ***********************************
      hFitNew, indexBestOffspring, indexWorstOffspring, coefArrNew, intercepArrNew, cuModelNew = gpM1.ComputeError(self,
              hOutIndividuals, 
              self.y_train, 
              self.Individuals, 
              self.nrowTrain,
              hStack, 
              hStackIdx,
              self.evaluationMethod)

      oldFit = hFit[indexBestIndividual_p]
      newFit = hFitNew[indexBestOffspring]

      #print("hFit:", hFit[indexBestIndividual_p], " indexBestIndividual_p:", indexBestIndividual_p)
      #print("hFitNew:", hFitNew[indexBestOffspring], " indexBestOffspring:", indexBestOffspring)

      # *********************** FUNTIONS WEIGHT EVALUATION ***********************

      # Mejor individuo de la nueva generacion (BestOffspring)
      idx_a1 = indexBestOffspring * self.GenesIndividuals
      idx_b1 = indexBestOffspring * self.GenesIndividuals + self.GenesIndividuals
      mBestIndividual = hNewPopulation[idx_a1:idx_b1]


      #Initialize operators weigth
      #pesos_por_id = {op: 1.0/len(self.valid_functions_set) for op in self.valid_functions_set}
      pesos_por_id = gpM2.actualizar_pesos_operadores(pesos_por_id, mBestIndividual, oldFit, newFit, self.valid_functions_set)
      op_weights = np.array([pesos_por_id[int(oid)] for oid in self.valid_functions_set], dtype=np.float32)

      # print(mBestIndividual)
      # print(self.valid_functions_set)
      #print(pesos_por_id)
      #print(op_weights)
      gpM2.print_pesos_ordenados(pesos_por_id, self.valid_functions_set, gpG.OPERADOR_POR_ID)

      # Prepara la CDF (Cumulative Distribution Function) una vez por generación
      cdf = gpM2.preparar_operadores_numba(
          op_ids=self.valid_functions_set, op_weights=op_weights,
          epsilon=0.02, temperatura=1.0
      )
      #print(cdf)

      # *********************** NEW SURVIVAL (Elitist) ***********************
      hNewPopulation, indexBestIndividual_p, coefArr_p, intercepArr_p, cuModel_p, stackBestModel_p = gpM1.Survival(self,
              indexBestIndividual_p,
              indexBestOffspring,
              indexWorstOffspring,
              hInitialPopulation,
              hNewPopulation,
              hFit,
              hFitNew,
              coefArr_p, 
              intercepArr_p, 
              cuModel_p,
              stackBestModel_p,
              coefArrNew,
              intercepArrNew,
              cuModelNew,
              stackBestModelNew)
      # *********************** END NEW SURVIVAL ***********************

      # ***********************    NEW REPLACE   ***********************
      hInitialPopulation, hFit = gpM1.replace(self,
                      hInitialPopulation,
                      hNewPopulation, 
                      hFit,
                      hFitNew)
      # *********************** END NEW REPLACE ***********************
      # print (hInitialPopulation)
      

			# Validate Best Individual with Test file for generation
			#/*trainFit = checkFitness(config, handle, dataFile, dInitialPopulation, indexBestIndividual_p, 0);*/
      ajFit = 0
      if (self.evaluationMethod == 1) : #or (self.scorer == 2) : 
        ajFit = gpG.MAX_R2_NEG * (-1)

      bestFitGeneration = hFit[indexBestIndividual_p]
      trainFit = hFit[indexBestIndividual_p] - ajFit

      # Obtenemos la longitud del stack del mejor papa
      BestIndividualLength = gpG.bestIndividualInfo(self, hInitialPopulation,  indexBestIndividual_p)
     

      if self.verbose == 1 :
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print("Generation:",generation, " Best Index:",indexBestIndividual_p, " Length Indiv:",BestIndividualLength, " Train Fit:",trainFit, f" Time lapsed:{elapsed}")
			#end if

      del hStack
      del hStackIdx
      del hStackModel
      del hOutIndividuals 
      gc.collect()
      
      
      if hFitNew[indexBestIndividual_p] <= 0.0000000000000001 :
        break

    #end for 
    # ************* Fin de for (Ciclo Generacional) ****************

    # print(mBestIndividual)
    # print("Operadores validos:")
    # print(self.valid_functions_set)
    # print("Pesos por id:")
    # print(pesos_por_id)
    # print("Pesos:")
    # print(op_weights)

    # Obtenemos el mejor individuo
    idx_a1 = indexBestIndividual_p * self.GenesIndividuals
    idx_b1 = indexBestIndividual_p * self.GenesIndividuals + self.GenesIndividuals
    self.bestIndividual = hInitialPopulation[idx_a1:idx_b1]
    self.model = self.bestIndividual

    #print("Index bestIndividual:")
    #print(indexBestIndividual_p)
    #print("bestIndividual:")
    #print(self.bestIndividual)

    # Para caso de evaluaciones utilizando cuML se construye una 
    # expresion utilizando todas expresiones del stack que generan 
    # la matriz semantica 
    if (self.evaluationMethod >= 2 ) :
      self.cuModel = copy.deepcopy(cuModel_p)
      #self.maxRandomConstant = gpG.MAX_CONSTANT

      #print("X_train:")
      #print(X_train)

      #X_train2 = X_train[0]
      #print("X_train2:")
      #print(X_train2)

      #sacamos el mejor modelo del stack de expresiones 
      stackBestModel_p = gpM1.getStackBestModel(
                  self.bestIndividual,
                  X_train,
                  self.Individuals,
                  self.GenesIndividuals,
                  self.nrowTrain,
                  self.nvar) 
    
      #print("stackBestModel_p:")
      #print(stackBestModel_p)

      # Se construye una nueva pila con las todas expresiones 
      # generadas y almacenadas en el stack del mejor modelo
      allStackExpr = gpG.getStackModelExpr(self, stackBestModel_p)
      
      #print("allStackExpr:")
      #print(allStackExpr)

      # De la cadena completa de expresiones obtenemos el numero  
      # de stacks de expresiones disponibles
      # 'X:Y:Z:<Expr1>', 'X:Y:Z:<Expr2>', .... , 'X:Y:Z:<ExprN'  
      # (0)X=Total de elementos, 
      # (1)Y=Numero de stacks, 
      # (2)Z=Elemento de este stack 
      tmpModelExpr = allStackExpr[0]
      tmp = tmpModelExpr.split(':')
      nStack = int(tmp[1])

      # Se crea una nueva pila para guardar las expresiones de cada 
      # elememento del stack y posteriormente formar el modelo tipo M4GP
      nvoModel = [] 
      m4gpModel = gpG.m4gpModel(self, stackBestModel_p, 
                                coefArr_p, 
                                intercepArr_p) 
      
      #print("m4gpModel:")
      #print(m4gpModel)

      #print("self.cuModel.coef_.shape:", self.cuModel.coef_.shape)
      #print("self.cuModel.coef_:", self.cuModel.coef_)

      # Reconstruccion del modelo.
      # Se agregan los coeficientes obtenidos del modelo de evaluacion cuML
      # Por cada expresion, tenemos un coefiente  
      
      for j in range(nStack):
        #Obtenemos las expresiones del stack
        tmp1 = m4gpModel.get()
        tmp2 = coefArr_p
        tmp3 = tmp2[nStack-j-1]
        if(math.isnan(tmp3) or math.isinf(tmp3)) :
          tmp3 = 0

        #print(tmp1)
        #print(tmp2)
        #print(tmp3)

        # Solo interesan expresiones cuyo coeficiente no sea cero
        if (tmp3 != 0) :
          #Se agrega el coeficiente al inicio de la expresion
          #Se le agrega un operador de multiplicacion (*)
          nvoModel.insert(0,float(-10003)) 
          nvoModel.insert(0,float(tmp3))
          nvoModel =  gpG.m4gpBuildExpr(tmp1, nvoModel)
          if (j >= 1):
            nvoModel.append(-10001)            
      #end for

      tmp3 = intercepArr_p
      if (math.isnan(float(tmp3)) or (math.isinf(float(tmp3)))) :
        tmp3 = 0

      # Solo interesan expresiones cuyo coeficiente no sea cero
      if (tmp3 != 0) :
      #insertamos el intercept a la expresion
        nvoModel.append(float(tmp3))
        nvoModel.append(-10001)  # Se agrega un operador de suma (+)
      
      nvoModel.append(-11111)
      self.m4gpModel = np.array(nvoModel)
      del nvoModel
    #end if

    #Free local memory
    del hFit
    del hFitNew
    del hInitialPopulation

    # Clear lists
    gpCuM.coefArr.clear()
    gpCuM.intercepArr.clear()
    gpCuM.cuModel.clear()
    gc.collect()
    
    print("Finished Fit.")
    return	 
  # Fin de def (fit)

  def predict(self, X_predict):
    """
    Genera predicciones utilizando el modelo entrenado de M5GP.

    Este método aplica la transformación simbólica aprendida por el mejor
    individuo para construir la matriz semántica correspondiente a los datos
    de entrada. La semántica resultante es procesada por el modelo lineal
    previamente ajustado (cuML LinearRegression), produciendo así las
    predicciones finales.

    Parámetros
    ----------
    X : array-like, cuDF.DataFrame o cupy.array
        Matriz de características sobre la cual se desean generar predicciones.
        Debe contener el mismo número de columnas y el mismo orden que X_train
        utilizado durante el ajuste del modelo. Dimensión esperada:
        (n_muestras, n_caracteristicas).

    Retorna
    -------
    y_pred : array-like o cupy.array
        Vector con las predicciones generadas por el modelo. Su dimensión es:
        (n_muestras, ).

    Notas
    -----
    - Se utiliza el mejor individuo obtenido durante `fit()` y su semántica
      como transformación simbólica del espacio de características.
    - La generación de la semántica se ejecuta en GPU mediante kernels Numba.
    - El modelo lineal utilizado para predecir es el mismo que se ajustó
      durante el entrenamiento.
    - Es necesario haber llamado a `fit()` antes de ejecutar este método.
    """
  
    if (len(self.bestIndividual) == 0) :
      print("No model available for predict")
      return
    
    print("Inicio predict: ", X_predict.shape)

    self.X_predict = X_predict

    # Get number of data rows for predict
    self.nrowPredict = self.X_predict.shape[0]
    hDataPredict = np.reshape(self.X_predict, -1)

    
    numIndividuals = 1
    hModelPopulation = self.bestIndividual  
    GenesIndiv = hModelPopulation.shape[0] # self.GenesIndividuals

    # ***************************  Compute Individuals  ****************************
    hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
            hModelPopulation,
            hDataPredict,
            numIndividuals,
            GenesIndiv,
            self.nrowPredict,
            self.nvar,
            0 )

    y_pred=[]

    stackBestModel_p = gpM1.getStackBestModel(
                hModelPopulation,
                self.X_predict,
                numIndividuals,
                GenesIndiv,
                self.nrowPredict,
                self.nvar) 
    #allModelExpr = gpG.getModelExpr(self, stackBestModel_p)   

    if (self.evaluationMethod < 2 ) :
      for i in range(self.nrowPredict):
        y_pred.append(hOutIndividuals[i])

      y_pred = np.array(y_pred)
    else :
      st = hStack.reshape(numIndividuals, self.nrowPredict * GenesIndiv)
      ind = st[0]
      ind2 = ind.reshape(self.nrowPredict, GenesIndiv)
      tt = int(hStackIdx[0])
      
      sX_train = ind2[:, :tt]     
      cX = cp.asarray(sX_train, dtype=cp.float64)
      y_predModel = self.cuModel.predict(cX)
      y_pred = cp.asnumpy(y_predModel)

      #Free local memory
      del st
      del ind
      del ind2
      del tt
      del sX_train
      del cX
      del y_predModel
    #End if

    #Free local objects memory
    del hStack
    del hStackIdx
    del hStackModel
    del hOutIndividuals 
    del hDataPredict
    del hModelPopulation
    gc.collect()

    return y_pred
  # Fin de def (predict)

  # def getModelExpr(self, model):
  #   allModelExpr = gpG.getStackModelExpr(self, model) 

  #   print(allModelExpr)
  #   tmpModelExpr = allModelExpr[0]
  #   tmp = tmpModelExpr.split(':')
  #   nStack = int(tmp[1])

  #   BestModelExpr = allModelExpr[nStack-1]
  #   tmp = BestModelExpr.split(':')
  #   indivLenght = tmp[0]
  #   nStack = tmp[1]
  #   complexity = tmp[2] 
  #   modelExpr = tmp[3]

  #   return modelExpr
  
  def best_individual(self):
    if ((self.model == 0).all()) :
      print("No model available")
      return
    
    if (self.evaluationMethod < 2 ) :
      model = self.model
    else :
      model = self.m4gpModel

    allModelExpr = gpG.getStackModelExpr(self, model) 

    tmpModelExpr = allModelExpr[0]
    tmp = tmpModelExpr.split(':')
    nStack = int(tmp[1])

    BestModelExpr = allModelExpr[nStack-1]
    tmp = BestModelExpr.split(':')
    indivLenght = tmp[0]
    nStack = tmp[1]
    complexity = tmp[2] 
    modelExpr = tmp[3]

    return modelExpr  
  # Fin de def (best_individual) 

  def get_model(self):
    return self.best_individual()
  # Fin de def (get_model) 

  def get_n_nodes(self):
    if (self.evaluationMethod < 2 ) :
      model = self.model
    else :
      model = self.m4gpModel     

    allModelExpr = gpG.getStackModelExpr(self, model) 
    tmpModelExpr = allModelExpr[0]
    tmp = tmpModelExpr.split(':')
    nStack = int(tmp[1])

    BestModelExpr = allModelExpr[nStack-1]

    tmp = BestModelExpr.split(':')
    nStack = tmp[1]
    nodes = tmp[2] 

    return str(nodes)
  # Fin de def (get_n_nodes) 

  def complexity(self):
    return self.get_n_nodes()
  # Fin de def (complexity) 
   
  def meanSquaredError(self, cY, YPred) :
    # if (len(cY) == 0 or type(YPred == 'NoneType') or len(YPred) ==0):
    #   print("Not cY or YPred providen")
    #   return
    
    npY = np.array(cY).astype('float32')

    npYPred = YPred
    #mse = mean_squared_error(npY, npYPred, squared=False)
    mse = mean_squared_error(npY, npYPred)
    return mse

  def rmse(self, cY, YPred) :
    # if (len(cY) == 0 or type(YPred == 'NoneType') or len(YPred) ==0):
    #   print("Not cY or YPred providen")
    #   return
    
    mse = self.meanSquaredError(cY, YPred) 
    mse = math.sqrt(mse)
    return mse
   		
  def R2(self, cY, YPred):
    # if (len(cY) == 0 or type(YPred == 'NoneType') or len(YPred) ==0):
    #   print("Not cY or YPred providen")
    #   return
    
    r2 = r2_score(cY, YPred)
    return r2
  
  def getStackExpr(self, Model) :
    self.nvar=7
    allModelExpr = gpG.getStackModelExpr(self, Model)
    print(allModelExpr)
    return
  

"""
    M5GP Clasificador. Modelo de programación genética combinado con un modelo de cuML para clasificacion.
"""
class m5gpClassifier(BaseEstimator):
  """
    Modelo de programación genética combinado con un modelo de cuML.

    Parámetros
    ----------
    generations : int, opcional, default=2
      Número de generaciones (limitado por defecto).

    Individuals : int, opcional, default=16
      Número de individuos.

    GenesIndividuals : int, opcional, default=256
      Número de genes por individuo.

    mutationProb : float, opcional, default=0.1
      Probabilidad de tasa de mutación.

    mutationDeleteRateProb : float, opcional, default=0.01
      Probabilidad de tasa de eliminación de mutación.

    sizeTournament : float, opcional, default=0.15
      Tamaño del torneo.

    evaluationMethod : int, opcional, default=0
      Modelo de machine learning que se utiliza en combinación con la programación genética para la evaluación de error.
      Opciones:
        - 0: Logistic Regression
        - 1: Support Vector Classifier
        - 2: Random Forest Classifier
        - 3: K Neighbors Classifier

    scorer : int, opcional, default=2
    Métrica utilizada para evaluar el desempeño del modelo de machine learning en combinación con la programación genética.
    
    Opciones:
        - 0: Accuracy Score (accuracy de cuML)
        - 1: ROC AUC Score (cuROCAUC de cuML)
        - 2: F1 Score (f1_score de scikit-learn)
        - 3: Average Precision Score (average_precision_score de scikit-learn)

    averageMode : str, opcional, default='macro'
    Solo disponible para F1 Score & Average Precision Score

    Opciones:
        - 'micro': Calcula la puntuación F1 globalmente contando el total de verdaderos positivos, falsos negativos y falsos positivos.
        - 'macro': Calcula la puntuación F1 para cada clase y luego calcula la media sin ponderar de estas puntuaciones.
        - 'weighted': Calcula la puntuación F1 para cada clase y luego calcula la media ponderada de estas puntuaciones según el soporte de cada clase.
        - 'samples': Calcula la puntuación F1 para cada instancia y luego calcula la media de estas puntuaciones.
        

    maxRandomConstant : int, opcional, default=999
      Número de constantes (-maxRandomConstant a maxRandomConstant).

    genOperatorProb : float, opcional, default=0.50
      Probabilidad de generar operadores.

    genVariableProb : float, opcional, default=0.39
      Probabilidad de generar variables.

    genConstantProb : float, opcional, default=0.1
      Probabilidad de generar constantes.

    genNoopProb : float, opcional, default=0.01
      Probabilidad de generar operadores NOOP.

    useOpIF : int, opcional, default=0
      Establece si se utiliza el operador IF.

    verbose : int, opcional, default=1
      Muestra mensajes durante la ejecución.

    crossVal : bool, opcional, default=True
      Indica si se realiza validación cruzada.

    k : int, opcional, default=5
      Número de divisiones para la validación cruzada.
  """
  #method to initialize the class
  def __init__(self, 
            generations=1, 
            Individuals=3, 
            GenesIndividuals=1024, 
            mutationProb=0.1, 
            mutationDeleteRateProb=0.01,  
            evaluationMethod=0, 
            scorer=0,  
            sizeTournament=0.15, 
            maxRandomConstant=1, 
            genOperatorProb=0.50, 
            genVariableProb=0.39, 
            genConstantProb=0.1, 
            genNoopProb=0.01,  
            useOpIF=0,   
            log=1, 
            functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "sqrt", "exp", "log", "abs"], 
            verbose=1, 
            logPath='log/',
            function_set = '',
            crossVal = True,
            k = 3 ,
            averageMode = "macro",
            CrossAverage = False,
            params=None,
            **kwargs):

    env = dict(os.environ)
    self.generations = generations
    self.Individuals=Individuals 
    self.GenesIndividuals=GenesIndividuals
    self.mutationProb=mutationProb 
    self.mutationDeleteRateProb=mutationDeleteRateProb  
    self.evaluationMethod=evaluationMethod
    self.scorer = scorer
    self.sizeTournament=sizeTournament 
    self.maxRandomConstant=maxRandomConstant 
    self.genOperatorProb=genOperatorProb 
    self.genVariableProb=genVariableProb 
    self.genConstantProb=genConstantProb 
    self.genNoopProb=genNoopProb  
    self.useOpIF=useOpIF
    self.functions_set = functions_set
    self.nvar=0 
    self.nrowTrain=0 
    self.nrowTest=0
    self.log=log
    self.verbose=verbose
    self.logPath=logPath
    self.function_set=function_set
    self.model = ''
    self.m4gpModel = ''
    self.cuModel = 0
    self.bestIndividual = ''
    self.crossVal = crossVal
    self.k = k
    self.averageMode = averageMode
    self.params = None
    self.CrossAverage = CrossAverage

    #print(gpCuM2.cuGetMethodNameClassification(self))
    print("Initializing m5gp classifier")

    # Check if CUDA and Device GPU are available
    if not gpG.cudaSetup(0):
      print("Check CUDA device. Fail to initialize.")
      return

    # Verifica los operadores validos y construye el diccionario a utilizar
    # para generar la poblacion inicial
    if (len(self.functions_set) == 0):
      print("No se definieron operadores")
      exit(0)
    
    # Get valid functions (mathematical operators) allowed for generate individuals
    self.valid_functions_set = gpG.construir_lista_operadores_validos(self.functions_set)
    if (len(self.valid_functions_set) == 0):
      print("No se especificaron operadores validos")
      exit(0)
  

    fName = "M5GP_OpS.csv"
    if os.path.exists(fName):
        os.remove(fName)
    return


  '''
    Ejecuta el proceso completo de generación y entrenamiento del modelo M5GP para clasificacion.
  '''
  #This method implement the evolution with M5GP  
  def fit(self, X_train, y_train):
    # Se validan los valores asignados para las probabilidades de generacion de
    # Funciones matematicas, Variables, constantes y operadores NOOP
    # Se normalizan la probabilidades por seguridad
    totalProb = self.genOperatorProb + self.genVariableProb + self.genConstantProb + self.genNoopProb
    if totalProb <= 0.0:
          # fallback razonable
      p_op_n, p_var_n, p_const_n, p_noop_n = 0.53, 0.37, 0.05, 0.05
    else:
      inv = 1.0 / totalProb
      p_op_n, p_var_n, p_const_n, p_noop_n = self.genOperatorProb*inv, self.genVariableProb*inv, self.genConstantProb*inv, self.genNoopProb*inv

    # Se reasignan las probabilidades normalizadas
    self.genOperatorProb=p_op_n 
    self.genVariableProb=p_var_n 
    self.genConstantProb=p_const_n 
    self.genNoopProb=p_noop_n
    #self.maxRandomConstant= np.float32(self.maxRandomConstant)   

    self.X_train = X_train
    self.y_train = y_train

    # train data    
    data=pd.DataFrame(self.X_train)
    data['target']=self.y_train

    self.nrowTrain = len(data.index)
    self.nrowTest = len(data.index)
    self.nrowPredict = len(data.index)
    self.nvar = data.shape[1] - 1
 
    print("Executing Fit - Method(", self.evaluationMethod ,") - ", gpCuM2.cuGetMethodNameClassification(self), " Scorer:", self.scorer)
    print("nRows:", self.nrowTrain, "nVars:", self.nvar)

    #Initialize operators weigth
    pesos_por_id = {op: 1.0/len(self.valid_functions_set) for op in self.valid_functions_set}
    op_weights = np.array([pesos_por_id[int(oid)] for oid in self.valid_functions_set], dtype=np.float32)
      
    # Prepara la CDF (Cumulative Distribution Function) una vez por generación (Numba)
    # CDF => lista de probabilidades acumuladas que se usa para hacer selección aleatoria ponderada.
    # Esto te permite hacer selección basada en probabilidad directamente.
    cdf = gpM2.preparar_operadores_numba(
        op_ids=self.valid_functions_set, op_weights=op_weights,
        epsilon=0.02, temperatura=1.0
    )
    
    # print("Pesos iniciales")
    # print(pesos_por_id)
    # print(op_weights)
    # print(cdf)

    # Store the size in bytes for initial population
    gpG.sizePopulation = self.Individuals * self.GenesIndividuals 
    gpG.sizeIndividuals = self.Individuals 
    gpG.sizeTournament = math.ceil(self.sizeTournament * self.Individuals)

    # Define vectors to work on device 
    self.model = np.zeros((self.GenesIndividuals ), dtype=np.float32) 

    #print("Initialize Individual")
    # *************************** Initialize population ********************************* 
    hInitialPopulation = gpM1.initialize_population(
                              self.Individuals,
                              self.nvar,
                              self.GenesIndividuals,
                              self.maxRandomConstant,
                              self.genOperatorProb,
                              self.genVariableProb,
                              self.genConstantProb,
                              self.genNoopProb,
                              self.useOpIF,
                              self.valid_functions_set,
                              cdf )
    # -- End of Initialize population --

    # print("Individuals:")
    # print(hInitialPopulation)
    # return
  
    # ***************************  Compute Individuals  ****************************
    hOutIndividuals = [] 
    hStack = []
    hStackIdx = []
    hStackModel = []
  

    #print ("Compute Individual")
    hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
            hInitialPopulation,
            self.X_train,
            self.Individuals,
            self.GenesIndividuals,
            self.nrowTrain,
            self.nvar,
            0 )
    # ****************** End of Compute Individuals **********************
    
    # Get the semantic matrix
    coefArr_p = []
    intercepArr_p = []    
    cuModel_p = []  
    stackBestModel_p = []

    coefArrNew = []
    intercepArrNew = [] 
    cuModelNew = []
    stackBestModelNew = []

    hFit = np.zeros((gpG.sizeIndividuals), dtype=np.float32)
    hFitNew = np.zeros((gpG.sizeIndividuals), dtype=np.float32)
    indexBestOffspring = 0
    indexWorstOffspring = 0


    #print("Compute Error")
    # ***************************** Compute ERROR ***********************************
    hFit, indexBestOffspring, indexWorstOffspring, cuModel_p = gpM3.ComputeErrorClassification(self,
                hOutIndividuals, 
                y_train, 
                self.Individuals, 
                self.nrowTrain,
                hStack, 
                hStackIdx,
                self.evaluationMethod)

    # Index of best individual of initialization
    indexBestIndividual_p = indexBestOffspring 

    del hStack
    del hStackIdx
    gc.collect()

    ajFit = 0
    trainFit = hFit[indexBestIndividual_p] - ajFit    
    print("Initial Index:", indexBestIndividual_p, " Initial Fit:", trainFit)


    # ***********************************************************************
    # ********************* GP Process Generation Cycle *********************
    # ***********************************************************************
    print ("Starting generational process")
    for generation in range(1,self.generations + 1):
      trainFit = 0
      testFit = 0
      coefArrNew = []
      intercepArrNew = [] 
      cuModelNew = []
      stackBestModelNew = []
      start_time = time.time()

      #print("Torneo")
      # *********************  Select Tournament  **********************
      hNewPopulation, hBestParentsTournament = gpM1.select_tournament(
                    hInitialPopulation,
                    hFit,
                    self.Individuals, 
                    self.GenesIndividuals )

      #print("Mutacion")
      # *********************  UMAD Mutation  **********************
      hNewPopulation = gpM1.umadMutation(self,
                                  hInitialPopulation,
                                  hBestParentsTournament,
                                  self.Individuals,
                                  cdf) 

      #print (hNewPopulation)
      # ***************************  Compute Individuals  ****************************
      hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
              hNewPopulation,
              self.X_train,
              self.Individuals,
              self.GenesIndividuals,
              self.nrowTrain,
              self.nvar,
              0 )
      
      # ***************************** Compute ERROR ***********************************
      hFitNew, indexBestOffspring, indexWorstOffspring, cuModelNew = gpM3.ComputeErrorClassification(self,
              hOutIndividuals, 
              y_train, 
              self.Individuals, 
              self.nrowTrain,
              hStack, 
              hStackIdx,
              self.evaluationMethod)

      oldFit = hFit[indexBestIndividual_p]
      newFit = hFitNew[indexBestOffspring]

      #print("hFit:", hFit[indexBestIndividual_p], " indexBestIndividual_p:", indexBestIndividual_p)
      #print("hFitNew:", hFitNew[indexBestOffspring], " indexBestOffspring:", indexBestOffspring)

      # Mejor individuo de la nueva generacion (BestOffspring)
      idx_a1 = indexBestOffspring * self.GenesIndividuals
      idx_b1 = indexBestOffspring * self.GenesIndividuals + self.GenesIndividuals
      mBestIndividual = hNewPopulation[idx_a1:idx_b1]

      # *********************** FUNTIONS WEIGHT EVALUATION ***********************
      pesos_por_id = gpM2.actualizar_pesos_operadores(pesos_por_id, mBestIndividual, oldFit, newFit, self.valid_functions_set)
      op_weights = np.array([pesos_por_id[int(oid)] for oid in self.valid_functions_set], dtype=np.float32)


      # Prepara la CDF (Cumulative Distribution Function) una vez por generación
      cdf = gpM2.preparar_operadores_numba(
          op_ids=self.valid_functions_set, op_weights=op_weights,
          epsilon=0.02, temperatura=1.0
      )

      # *********************** NEW SURVIVAL (Elitist) ***********************
      hNewPopulation, indexBestIndividual_p, cuModel_p, stackBestModel_p = gpM3.SurvivalClassification(self,
              indexBestIndividual_p,
              indexBestOffspring,
              indexWorstOffspring,
              hInitialPopulation,
              hNewPopulation,
              hFit,
              hFitNew,
              cuModel_p,
              stackBestModel_p,
              cuModelNew,
              stackBestModelNew)
      # *********************** END NEW SURVIVAL ***********************

      # ***********************    NEW REPLACE   ***********************
      hInitialPopulation, hFit = gpM1.replace(self,
                      hInitialPopulation,
                      hNewPopulation, 
                      hFit,
                      hFitNew)
      # *********************** END NEW REPLACE ***********************
      # print (hInitialPopulation)
      

			# Validate Best Individual with Test file for generation
			#/*trainFit = checkFitness(config, handle, dataFile, dInitialPopulation, indexBestIndividual_p, 0);*/
     
      ajFit = 0
      bestFitGeneration = hFit[indexBestIndividual_p]
      trainFit = hFit[indexBestIndividual_p] - ajFit

      # Obtenemos la longitud del stack del mejor papa
      BestIndividualLength = gpG.bestIndividualInfo(self, hInitialPopulation,  indexBestIndividual_p)
     

      if self.verbose == 1 :
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print("Generation:",generation, " Best Index:",indexBestIndividual_p, " Length Indiv:",BestIndividualLength, " Train Fit:",trainFit, f" Time lapsed:{elapsed}")
			#end if

      del hStack
      del hStackIdx
      del hStackModel
      del hOutIndividuals 
      gc.collect()
      
      
      if hFitNew[indexBestIndividual_p] <= 0.0000000000000001 :
        break

    #end for 
    # ************* Fin de for (Ciclo Generacional) ****************


    # Obtenemos el mejor individuo
    idx_a1 = indexBestIndividual_p * self.GenesIndividuals
    idx_b1 = indexBestIndividual_p * self.GenesIndividuals + self.GenesIndividuals
    self.bestIndividual = hInitialPopulation[idx_a1:idx_b1]
    self.model = self.bestIndividual

    #print("Index bestIndividual:")
    #print(indexBestIndividual_p)
    #print("bestIndividual:")
    #print(self.bestIndividual)

    # Para caso de evaluaciones utilizando cuML se construye una 
    # expresion utilizando todas expresiones del stack que generan 
    # la matriz semantica 

    self.cuModel = copy.deepcopy(cuModel_p)

    #sacamos el mejor modelo del stack de expresiones 
    stackBestModel_p = gpM1.getStackBestModel(
                self.bestIndividual,
                X_train,
                self.Individuals,
                self.GenesIndividuals,
                self.nrowTrain,
                self.nvar) 
  
    # Se construye una nueva pila con las todas expresiones 
    # generadas y almacenadas en el stack del mejor modelo
    allModelExpr = gpG.getStackModelExpr(self, stackBestModel_p)

    print("allStackExpr:")
    print(allModelExpr)
  
    final_expression = []
    nodes = True
    for expression in allModelExpr:
      array = expression.split(":")
      if nodes:
        self.nodes = array[1]
        nodes = False

      final_expression.append(array[3])
    self.m4gpModel = final_expression

    #Free local memory
    del hFit
    del hFitNew
    del hInitialPopulation

    # Clear lists
    # gpCuM.coefArr.clear()
    # gpCuM.intercepArr.clear()
    # gpCuM.cuModel.clear()
    gc.collect()
    
    print("Finished Fit.")
    return	 
  # Fin de def (fit)


  def predict(self, X_predict, probability=False):
    print("Inicio predict: ", X_predict.shape)
    

    # Get number of data rows for predict
    self.nrowPredict = X_predict.shape[0]
    hDataPredict = np.reshape(X_predict, -1)
    numIndividuals = 1
    hModelPopulation = self.bestIndividual  
    GenesIndiv = hModelPopulation.shape[0] # self.GenesIndividuals

    # ***************************  Compute Individuals  ****************************
    hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
            hModelPopulation,
            hDataPredict,
            numIndividuals,
            GenesIndiv,
            self.nrowPredict,
            self.nvar,
            0 )

    y_pred = []

  
    st = hStack.reshape(numIndividuals, self.nrowPredict * GenesIndiv)
    ind = st[0]
    ind2 = ind.reshape(self.nrowPredict, GenesIndiv)
    tt = int(hStackIdx[0])
    
    sX_train = ind2[:, :tt]     
    cX = cp.asarray(sX_train, dtype=cp.float64)
    if probability == True:
      y_predModel = self.cuModel.predict_proba(cX)
    else:
      y_predModel = self.cuModel.predict(cX)
    y_pred = cp.asnumpy(y_predModel)

    if X_predict.shape == (2,):
      y_pred = y_pred[0]

    #Free local memory
    del st
    del ind
    del ind2
    del tt
    del sX_train
    del cX
    del y_predModel
    #End if

    #Free local objects memory
    del hStack
    del hStackIdx
    del hStackModel
    del hOutIndividuals 
    del hDataPredict
    del hModelPopulation
    gc.collect()

    return y_pred
  # Fin de def (predict)

  def predict_proba(self, X_predict):
    return self.predict(X_predict,True)

  def best_individual_expression(self):
    return  self.m4gpModel  
  # Fin de def (best_individual) 

  def complexity(self):
    return self.nodes
  # Fin de def (complexity) 
   
  def score(self, X, y, metric=0, averageMode="macro") :

    if metric == 3:
      y_pred = self.predict_proba(X)
    else: 
      y_pred = self.predict(X)

    score = gpCuM.evaluationMetrics(metric, y, y_pred, averageMode)

    return score
