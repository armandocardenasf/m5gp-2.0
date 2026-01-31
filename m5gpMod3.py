# *********************************************************************
# Name: m5gpMod3.py
# Description: Modulo que implementa metodos tipo wrapper para ejecutar
# metodos CUDA y CuML a traves de llamadas comunes
# Se implementa la logica de ejecucion para funciones de numba y CuML.
# *********************************************************************

import math
import time
import gc

try:
  import numpy as np
  import cupy as cp

  from numba import cuda
  from numba.cuda.random import (create_xoroshiro128p_states,
                                xoroshiro128p_uniform_float32)
  GPU_IMPORTS = True
except ImportError:
  GPU_IMPORTS = False  

import m5gpGlobals as gpG
import m5gpCudaMethods as gpCuda
import m5gpCumlMethods as gpCuM
import m5gpCumlMethods2 as gpCuM2



def ComputeErrorClassification(self,
                hOutIndividuals, 
                hDataY, 
                numIndividuals, 
                nrowTrain,
                hStack, 
                hStackIdx,
                evaluationMethod) :
    cuModel_p = []

    result_train_p = 0
    hFit = np.zeros((gpG.sizeIndividuals), dtype=np.float32)
    dFit = cuda.to_device(hFit)

    dOutIndividuals = cuda.to_device(hOutIndividuals)
    dDataY = cuda.to_device(hDataY)

    #gridsize = gpCuda.gpuMaxUseProc(self.Individuals, blocksize)
    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    blocksize = MaxOcup["BlockSize"]
    gridsize = MaxOcup["GridSize"] 

    start_time = time.time()
    
    if (evaluationMethod == 0 or #M4GP - 2=cuML LinearRegression
        evaluationMethod == 1 or #M4GP - 3=cuML Lasso regularization
        evaluationMethod == 2 or #M4GP - 4=cuML Ridge regularization
        evaluationMethod == 3 or #M4GP - 5=cuML kernel Ridge Regression
        evaluationMethod == 4 or #M4GP - 6=cuML Elasticnet regularization 
        evaluationMethod == 5 or #M4GP - 7=cuML MiniBatch none regularization
        evaluationMethod == 6 or #M4GP - 8=cuML MiniBatch lasso regularization
        evaluationMethod == 7 or #M4GP - 9=cuML MiniBatch ridge regularization
        evaluationMethod == 8) : #M4GP - 10=cuML MiniBatch elasticnet regularization

        #start_time = time.time()
        cuModel = []

        print("Ejecuta ComputeErrorClassification")
        hFit, cuModel = gpCuM2.EvaluateCuml2Classification(self, hStack, hStackIdx, hFit, hDataY)

        #elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        #print(f"Time cuML lapsed: {elapsed}")
  
        dFit = cuda.to_device(hFit)
        result_off = gpG.np.argmax(dFit)        
        indexBestOffspring = result_off

        result_w = gpG.np.argmin(dFit)	
        indexWorstOffspring = result_w      

        # Obtenemos los coeficientes y el modelo del 
        # mejor individuo generados por cuML
        cuModel_p = cuModel[indexBestOffspring]
     
    # end if (Evaluation methods)

    elapsed = time.time() - start_time
    Ops = (numIndividuals * nrowTrain)
    gpG.WriteCSV_OpS("compute_error", elapsed,Ops)    
 
    return hFit, indexBestOffspring,  indexWorstOffspring, cuModel_p
# *************************  End of Evaluate Individuals  *************************

# *********************  Survival Classification (Elitist)  *******************************
def SurvivalClassification(self,
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
            stackBestModelNew) :
    
    idx_a1 = indexWorstOffspring * self.GenesIndividuals
    idx_b1 = indexWorstOffspring * self.GenesIndividuals + self.GenesIndividuals
    idx_a2 = indexBestIndividual_p * self.GenesIndividuals
    idx_b2 = indexBestIndividual_p * self.GenesIndividuals + self.GenesIndividuals
        
        # Checamos si la nueva generacion es mejor que la anterior
    if (hFit[indexBestIndividual_p] > hFitNew[indexBestOffspring]) :
          # La nueva generacion no fue mejor que la anterior
          # Copia el mejor individuo de la anterior generacion  (idx_a2:idx_b2)
          # al lugar del peor individuo de la nueva generacion (idx_a1:idx_b1)
      hNewPopulation[idx_a1:idx_b1] = hInitialPopulation[idx_a2:idx_b2]

          # Ahora el peor hijo es el mejor padre
      hFitNew[indexWorstOffspring] = hFit[indexBestIndividual_p]
      indexBestIndividual_p = indexWorstOffspring

    else :
          # La nueva generacion fue mejor que la anterior
      indexBestIndividual_p = indexBestOffspring

          # Si ezs un metodo decuML, copiamos los coeficientes del
          # mejor individuo de la nueva generacion como papa para
          # la siguiente generacion
      
      cuModel_p = cuModelNew
      stackBestModel_p = stackBestModelNew
    # End if
                      
        # End if
    # End if
    return hNewPopulation, indexBestIndividual_p, cuModel_p, stackBestModel_p
# ***************************  End of Survival (Elitist)  *****************************

