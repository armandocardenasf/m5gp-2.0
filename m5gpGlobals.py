# *********************************************************************
# Name: m5gpGlobals.py
# Description: Modulo que implementa variables y metodos globales
# para su uso comun en todos los modulos el sistema.
# *********************************************************************
 
import math
import os
from pickle import TRUE
import csv
import sys
import warnings
import numpy as np
import atexit


import ctypes

try:
  from numba import cuda
  import torch
  GPU_IMPORTS = True
except ImportError:
  GPU_IMPORTS = False  


# from   pycuda.compiler import SourceModule
# import pycuda.driver as cuda
# import pycuda.gpuarray as gpuarray
# import pycuda.curandom as curandom
# import pycuda.tools as tools


#import skcuda.cublas as cublas
from   datetime import datetime
import pandas as pd
from   random import randint
import time
import gc
import csv
from queue import LifoQueue


# import pycuda.driver as pycuda
# from pycuda.tools import make_default_context, DeviceMemoryPool, clear_context_caches

VAR_INI = -1000 # Initial value for variables
OP_INI = -10000 # Initial value for operators
OP_ADD = -10001 # Addition operator
OP_SUB = -10002 # Subtraction operator
OP_MUL = -10003 # Multiplication operator
OP_DIV = -10004 # Division operator
OP_SIN = -10005 # Sine operator
OP_COS = -10006 # Cosine operator
OP_EXP = -10007 # Exponential operator
OP_LOG = -10008 # Logarithm operator
OP_ABS = -10009 # Absolute value operator

OP_SUM = -10010 # Summation operator
OP_PRD = -10011 # Product operator
OP_AVG = -10012 # Average operator
OP_SDV = -10013 # Standard deviation operator

# ***************************************************************
# Si hay nuevos operadores/funciones, ponerlas en este espacio,
# entre el ultimo operador agregado y OP_END.
# ***************************************************************
OP_TAN = -10014 # Tan Operator
OP_TANH =  -10015 # Tan Hyperbolic Operator
OP_SQRT = -10016 # Square_Root Operator 

# ***************************************************************
# Si se agrega un nuevo operador, incrementar el valor de OP_END
# ***************************************************************
OP_END = -10017 # Final operator
OP_IF = OP_END

# ***************************************************************
# Operadores IF siempre deberan ser los ultimos de los operadores.
# ***************************************************************
OP_IFE = OP_END        # Conditional special operator equal to (if ==)
OP_IFG = OP_END + 1    # Conditional special operator greater than (if >)
OP_IFL = OP_END + 2    # Conditional special operator less than (if <)


OP_FIN = -11111  # Max number of operators
OP_NOOP = -10099  # Not Valid Operator

# Diccionario maestro de operadores y sus valores numéricos
OPERADORES_MASTER = {
    "+": OP_ADD, # Addition operator
    "-": OP_SUB, # Subtraction operator
    "*": OP_MUL, # Multiplication operator
    "/": OP_DIV, # Division operator
    "sin": OP_SIN, # Sine operator
    "cos": OP_COS, # Cosine operator
    "exp": OP_EXP, # Exponential operator
    "log": OP_LOG, # Logarithm operator
    "abs": OP_ABS, # Absolute value operator
    "sum": OP_SUM, # Summation operator
    "prd": OP_PRD, # Product operator
    "avg": OP_AVG, # Average operator
    "std": OP_SDV, # Standard deviation operator
    "tan": OP_TAN, # Tangent operator
    "tanh": OP_TANH, # Tangent hyperbolic operador"
    "sqrt": OP_SQRT, # Square_Root
    "if": OP_IF # Conditional operator
}

# Crear diccionario inverso ID → símbolo
OPERADOR_POR_ID = {v: k for k, v in OPERADORES_MASTER.items()}

PI = 3.14159265

MAX_R2_NEG  = -5000
MAX_RMSE = 9999999
MIN_RMSE = 0.0
MAX_CONSTANT = 999
MIN_CONSTANT = MAX_CONSTANT * (-1)


device_id = 0
gpu_memory = 0
free_mem = 0

sizePopulation = 0
sizeIndividuals = 0
sizeTournament = 0

# sizeMemPopulation = 0
# sizeMemIndividuals  = 0
# sizeTournament = 0

def get_gpu_memory_info():
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    cuda.cuMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    return free.value, total.value

def cudaSetup(gpu_device_number=0): 
    global device_id, gpu_memory, free_mem

    # Check if CUDA and Device GPU are available
    if torch.cuda.is_available():
        # Get the device name
        device = torch.cuda.get_device_name(gpu_device_number)
        device_id = device
        print(f"Using CUDA device: {device}")

        print("Initial memory info:")
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(gpu_device_number).total_memory
        gpu_memory = total_memory
        print(f"Total GPU memory: {total_memory / (1024**3):.2f} GB")  # Convert to GB
        
        # Get current memory allocation
        allocated_memory = torch.cuda.memory_allocated(gpu_device_number)
        print(f"Allocated GPU memory: {allocated_memory / (1024**3):.2f} GB")

        # Get cached memory
        cached_memory = torch.cuda.memory_reserved(gpu_device_number)
        print(f"Cached GPU memory: {cached_memory / (1024**3):.2f} GB")

        # Free up unused cached memory
        torch.cuda.empty_cache()
        print("Unused cached memory freed")

        # Get allocated memory after clearing cache
        allocated_memory_after = torch.cuda.memory_allocated(gpu_device_number)
        print(f"Allocated GPU memory after clearing cache: {allocated_memory_after / (1024**3):.2f} GB")

        free_mem = total_memory - allocated_memory_after
        print(f"Free GPU memory : { free_mem / (1024**3):.2f} GB")

        print("Succesfully initialized CUDA")
        return True
    else:
      print("CUDA is not available.")
      return False


# Function for garbage collection in CUDA
# def cuda_finish():
#     global context
#     context.pop()
#     from pycuda.tools import clear_context_caches
#     clear_context_caches()
#     print("Finishing up PYCUDA")
#     return

def WriteCSV_OpS(nFun, elapsed,Ops, fCreate=False) :
    return
    nOpS = Ops / elapsed
    fName = "M5GP_OpS.csv"
    if fCreate == True :
        if os.path.exists(fName):
            os.remove(fName)
    with open(fName, 'a', newline='') as file:
        writer = csv.writer(file)       
        writer.writerow([nFun, elapsed, Ops, nOpS])  
        file.close()  
    #print(f"{nFun} Time lapsed: {elapsed}, Ops: {Ops}, nOpS : {nOpS}")
    return

def Truncate(f, n) :
    return math.floor(f * 10 ** n) / 10 ** n

def construir_lista_operadores_validos(operadores_deseados):
       
    # Construir una lista solo con los operadores solicitados
    diccionario_resultado = [OPERADORES_MASTER[op] for op in operadores_deseados if op in OPERADORES_MASTER]
    return np.array(diccionario_resultado, dtype=np.int32)


def bestIndividualInfo(config,  
                        dInitialPopulation,  
                        indexBestIndividual_p) :

    BestIndividualLength = 0
    numOpNOOP = 0
    umOpIf = 0
    numVars = 0
    numConst = 0
    numOps = 0
    numOpSin = 0
    numOpCos = 0
    numOpExp = 0
    numOpLog = 0
    numOpAbs = 0
    Expr = ""

    for i in range(config.GenesIndividuals):
        gene = dInitialPopulation[indexBestIndividual_p * config.GenesIndividuals + i];

        if (gene != OP_NOOP):
            BestIndividualLength =  BestIndividualLength + 1
        # if (gene == NOOP) :
        #     numOpNOOP = numOpNOOP + 1
        # if (gene == -10001) :
        #     numOps = numOps + 1
        #     Expr = Expr + "+\t"
        # if (gene == -10002) :
        #     numOps = numOps + 1
        #     Expr = Expr + "-\t"
        # if (gene == -10003) :
        #     numOps = numOps + 1
        #     Expr = Expr + "*\t"
        # if (gene == -10004) :
        #     numOps = numOps + 1
        #     Expr = Expr + "/\t"
        # if (gene == -10005) :
        #     numOpSin = numOpSin + 1
        #     Expr = Expr + "sin\t"
        # if (gene == -10006) :
        #     numOpCos = numOpCos + 1
        #     Expr = Expr + "cos\t"
        # if (gene == -10007) :
        #     numOpExp = numOpExp + 1
        #     Expr = Expr + "exp\t"
        # if (gene == -10008) :
        #     numOpLog = numOpLog + 1
        #     Expr = Expr + "log\t"
        # if (gene == -10009) :
        #     numOpAbs = numOpAbs +1
        #     Expr = Expr + "abs\t"
        # if ((gene == -10010) or (gene == -10011) or (gene == -10012)) :
        #     numOpIf = numOpIf + 1
        #     Expr = Expr + "if\t"
        # if ((gene <= -1000) and (gene > -10000)) :
        #     numVars = numVars + 1
        #     Expr = Expr + "X"
        #     Expr = Expr + str((int)((gene+1000) * (-1)))
        #     Expr = Expr + "\t"
        # if ((gene >= (config.maxRandomConstant * (-1) )) and (gene <= config.maxRandomConstant)) :
        #     numConst = numConst + 1
        #     Expr = Expr + " "
        #     Expr = Expr + str(gene)
        #     Expr = Expr + "\t"
        # End if
    # End for
    return BestIndividualLength

# Obtiene la expresion del individuo
def getIndividualExpr(config,  
                        dInitialPopulation,  
                        indexBestIndividual_p) :
    BestIndividualLength = 0
    numOpNOOP = 0
    umOpIf = 0
    numVars = 0
    numConst = 0
    numOps = 0
    numOpSin = 0
    numOpCos = 0
    numOpExp = 0
    numOpLog = 0
    numOpAbs = 0
    numOpSum = 0
    numOpPrd = 0
    numOpAvg = 0
    numOpSdv = 0
    numOpTan = 0
    numOpTanH = 0
    numOpSqrt = 0
    Expr = ""

    var_ini = math.fabs(VAR_INI)
    maxVar = float((var_ini + config.nvar -1) * (-1))
    for i in range(config.GenesIndividuals):
        gene = dInitialPopulation[indexBestIndividual_p * config.GenesIndividuals + i];

        if (gene != OP_NOOP) :
            BestIndividualLength =  BestIndividualLength + 1
        elif (gene == OP_NOOP) :
            numOpNOOP = numOpNOOP + 1
        elif (gene == OP_ADD) :
            numOps = numOps + 1
            Expr = Expr + "+\t"
        elif (gene == OP_SUB) :
            numOps = numOps + 1
            Expr = Expr + "-\t"
        elif (gene == OP_MUL) :
            numOps = numOps + 1
            Expr = Expr + "*\t"
        elif (gene == OP_DIV) :
            numOps = numOps + 1
            Expr = Expr + "/\t"
        elif (gene == OP_SIN) :
            numOpSin = numOpSin + 1
            Expr = Expr + "sin\t"
        elif (gene == OP_COS) :
            numOpCos = numOpCos + 1
            Expr = Expr + "cos\t"
        elif (gene == OP_EXP) :
            numOpExp = numOpExp + 1
            Expr = Expr + "exp\t"
        elif (gene == OP_EXP) :
            numOpLog = numOpLog + 1
            Expr = Expr + "log\t"
        elif (gene == OP_ABS) :
            numOpAbs = numOpAbs +1
            Expr = Expr + "Abs\t"

        elif (gene == OP_SUM) :
            numOpSum = numOpSum +1
            Expr = Expr + "SUM\t"
        elif (gene == OP_PRD) :
            numOpPrd = numOpPrd +1
            Expr = Expr + "PROD\t"
        elif (gene == OP_AVG) :
            numOpAvg = numOpAvg +1
            Expr = Expr + "AVG\t"
        elif (gene == OP_SDV) :
            numOpSdv = numOpSdv +1
            Expr = Expr + "SDV\t"

        elif (gene == OP_TAN) :
            numOpTan = numOpTan +1
            Expr = Expr + "tan\t"
        elif (gene == OP_TANH) :
            numOpTanH = numOpTanH +1
            Expr = Expr + "tanh\t"
        elif (gene == OP_SQRT) :
            numOpSqrt = numOpSqrt +1
            Expr = Expr + "sqrt\t"
            

        elif ((gene == OP_IFE) or (gene == OP_IFG) or (gene == OP_IFL)) :
            numOpIf = numOpIf + 1
            Expr = Expr + "if\t"       
        #if ((gene <= -1000) and (gene > -10000)) :
        elif ((gene <= VAR_INI) and ((gene >= maxVar)) and (gene.is_integer())) :
            numVars = numVars + 1
            Expr = Expr + "X"
            Expr = Expr + str((int)((gene + var_ini) * (-1)))
            Expr = Expr + "\t"
        elif ((gene >= (MIN_CONSTANT )) and (gene <= MAX_CONSTANT)) :
            numConst = numConst + 1
            Expr = Expr + " "
            Expr = Expr + str(gene)
            Expr = Expr + "\t"
        else :
            numConst = numConst + 1
            Expr = Expr + " "
            Expr = Expr + str(gene)
            Expr = Expr + "\t"            
        #End if
    #End for

    return Expr

# Obtiene el gen como expresion
def getGeneExp(config, gene) :
    Expr = ""

    var_ini = math.fabs(VAR_INI)
    maxVar = float((var_ini + config.nvar -1) * (-1))
    if (gene == OP_ADD) :
        Expr += "+"
    elif (gene == OP_SUB) :
        Expr += "-"  
    elif (gene == OP_MUL) :
        Expr += "*"
    elif (gene == OP_DIV) :
        Expr += "/"
    elif (gene == OP_SIN) :
        Expr += "sin"
    elif (gene == OP_COS) :
        Expr += "cos"
    elif (gene == OP_EXP) :
        Expr += "exp"
    elif (gene == OP_LOG) :
        Expr += "log"
    elif (gene == OP_ABS) :
        Expr += "abs"
        

    elif (gene == OP_SUM) :
        Expr += "+"
    elif (gene == OP_PRD) :
        Expr += "*"
    elif (gene == OP_AVG) :
        Expr += "avg"
    elif (gene == OP_SDV) :
        Expr += "sdv"

    elif (gene == OP_TAN) :
        Expr += "tan"
    elif (gene == OP_TANH) :
        Expr += "tanh"
    elif (gene == OP_SQRT) :
        Expr += "sqrt"

    elif ((gene <= VAR_INI) and ((gene >= maxVar)) and (gene.is_integer())) :
        Expr += "X_"
        Expr += str(int(((gene+1000) * (-1))))
    elif ((gene >= MIN_CONSTANT) and (gene <= MAX_CONSTANT)) :
        Expr += str(gene)
    else :
        Expr += str(gene)
    
    return Expr

# Obtiene todo el stack con todas las expresiones completas del modelo 
def getStackModelExpr(config, Model) :
    lenIndiv = 0
    stackModel = LifoQueue()
    Expr = ""
    tmpExpr = ""

    var_ini = math.fabs(VAR_INI)
    maxVar = float((var_ini + config.nvar -1) * (-1))

    lenModel = len(Model)
    for i in range(lenModel):
        gene = Model[i]
        if (gene == OP_FIN) :
            break
        geneExpr = getGeneExp(config, gene)
        #print(geneExpr)
        #lenIndiv += 1

        # ********************************* Es una constante ************************************/
        if ((gene >= MIN_CONSTANT) and (gene <= MAX_CONSTANT)) : # Es una constante
            tmpExpr = "1:"
            tmpExpr += "("+str(geneExpr)+")"
            stackModel.put(tmpExpr)
            lenIndiv += 1

        # ********************************* Es una variable ************************************/
        elif ((gene <= VAR_INI) and ((gene >= maxVar)) and (gene.is_integer())) :  # Es una variable
            tmpExpr= "1:"
            tmpExpr += geneExpr
            stackModel.put(tmpExpr)
            lenIndiv += 1

        # ************ Es un operador de Suma,Resta,Division o Multiplicacion ******************/
        elif ((gene == OP_ADD) or (gene == OP_SUB) or (gene == OP_MUL) or (gene == OP_DIV)) :
            # Es Suma,Resta,Division o Multiplicacion
            if (not stackModel.empty()) :
                tmp = stackModel.get() #Obtenemos el ultimo elemento del stack
                strCont = tmp[0 : tmp.find(":")]
                if (strCont.isnumeric()) :
                    tmpT = tmp
                    cont1 = int(strCont)
                    tmp  = tmp[tmp.find(":") + 1 : len(tmp)]
                    if (not stackModel.empty()) :
                        tmp2 = stackModel.get()
                        strCont = tmp2[0 : tmp2.find(":")]
                        cont2 = int(strCont)
                        tmp2  = tmp2[tmp2.find(":") +1 : len(tmp2)]
                        tmpExpr= str(cont1 + cont2 + 1)
                        tmpExpr += ":("
                        tmpExpr += tmp
                        tmpExpr += geneExpr
                        tmpExpr += tmp2
                        tmpExpr += ")"
                        stackModel.put(tmpExpr)
                    else :
                        stackModel.put(tmpT)      

                    lenIndiv += 1         
                    #End if
                # End if
            # End if

        # ********* Es un operador de seno, coseno, exponente, logaritmo, absoluto, tangente, tangente hyperbolica ************/
        elif ((gene ==  OP_SIN) or (gene == OP_COS) or (gene == OP_EXP)  or (gene == OP_ABS) or (gene == OP_TAN) or (gene == OP_TANH) or (gene == OP_SQRT)) :
            if (not stackModel.empty()) :
                tmp = stackModel.get()
                strCont = tmp[0 : tmp.find(":")]
                if (strCont.isnumeric()) :
                    cont1 = int(strCont)
                    tmp  = tmp[tmp.find(":") +1 : len(tmp)]
                    tmpExpr= str(cont1 + 1)
                    tmpExpr += ":("
                    tmpExpr += geneExpr
                    tmpExpr += "("
                    tmpExpr += tmp
                    tmpExpr += ")"
                    tmpExpr += ")"
                    stackModel.put(tmpExpr)
                    lenIndiv += 1
                #end if
            # End if

        # ********* Es un operador de logaritmo  ************/
        elif ((gene == OP_LOG)) :
            if (not stackModel.empty()) :
                tmp = stackModel.get()
                strCont = tmp[0 : tmp.find(":")]
                if (strCont.isnumeric()) :
                    cont1 = int(strCont)
                    tmp  = tmp[tmp.find(":") + 1 : len(tmp)]
                    tmpExpr= str(cont1 + 2)
                    tmpExpr += ":("
                    tmpExpr += geneExpr 
                    tmpExpr += "(abs("
                    tmpExpr += tmp
                    tmpExpr += "))"
                    tmpExpr += ")"
                    stackModel.put(tmpExpr)

                    lenIndiv += 2
                #end if
            # End if

        # ********* Es un operador de sumatoria  ************/
        elif ((gene == OP_SUM)) :
            #print("Encontro OP_SUM")
            if (not stackModel.empty()) :
                tmpExpr = ""
                Expr = ""
                ContT = 0
                    
                while(stackModel.qsize() > 0 ) :
                    #print("Encontro expresion: ")
                    tmp = stackModel.get() # Se obtiene la ultima expresion de la pila de expresiones
                    #print(tmp)
                    strCont = tmp[0 : tmp.find(":")] # obtener el numero de elementos 
                    if (strCont.isnumeric()) :
                        cont1 = int(strCont)
                        ContT += cont1
                        tmp  = tmp[tmp.find(":") + 1 : len(tmp)] # Obtenemos solo la expresion
                        #tmpExpr= str(cont1 + 1)
                        #tmpExpr += ":( ("  
                        tmpExpr += "("                                             
                        tmpExpr += tmp
                        tmpExpr += ")"
                        if (stackModel.qsize() > 0):
                            tmpExpr += "+" # geneExpr
                            ContT += 1
                            lenIndiv += 1
                    #end if
                #end while
                #tmpExpr= str(ContT)
                Expr = str(ContT) + ":((" + tmpExpr + ")"
                #tmpExpr += ":( ("
                stackModel.put(Expr)
                #end if
            # End if

        # ********* Es un operador de producto  ************/
        elif ((gene == OP_PRD)) :
            #print("Encontro OP_MUL")
            if (not stackModel.empty()) :
                tmpExpr = ""
                Expr = ""
                ContT = 0
                    
                while(stackModel.qsize() > 0 ) :
                    #print("Encontro expresion: ")
                    tmp = stackModel.get() # Se obtiene la ultima expresion de la pila de expresiones
                    #print(tmp)
                    strCont = tmp[0 : tmp.find(":")] # obtener el numero de elementos 
                    if (strCont.isnumeric()) :
                        cont1 = int(strCont)
                        ContT += cont1
                        tmp  = tmp[tmp.find(":") + 1 : len(tmp)] # Obtenemos solo la expresion
                        #tmpExpr= str(cont1 + 1)
                        #tmpExpr += ":( ("  
                        tmpExpr += "("                                             
                        tmpExpr += tmp
                        tmpExpr += ")"
                        if (stackModel.qsize() > 0):
                            tmpExpr += "*" #getGeneExp(config, gene)
                            ContT += 1
                            lenIndiv += 1

                    #end if
                #end while
                #tmpExpr= str(ContT)
                Expr = str(ContT) + ":((" + tmpExpr + ")"
                #tmpExpr += ":( ("
                stackModel.put(Expr)
                #end if
            # End if

        # ********* Es un operador de promedio  ************/
        elif ((gene == OP_AVG)) :
            #print("Encontro OP_AVG")
            if (not stackModel.empty()) :
                tmpExpr = ""
                Expr = ""
                ContT = 0
                NumExpr = stackModel.qsize()

                while(stackModel.qsize() > 0 ) :
                    #print("Encontro expresion: ")
                    tmp = stackModel.get() # Se obtiene la ultima expresion de la pila de expresiones
                    #print(tmp)
                    strCont = tmp[0 : tmp.find(":")] # obtener el numero de elementos 
                    if (strCont.isnumeric()) :
                        cont1 = int(strCont)
                        ContT += cont1
                        tmp  = tmp[tmp.find(":") + 1 : len(tmp)] # Obtenemos solo la expresion
                        #tmpExpr= str(cont1 + 1)
                        #tmpExpr += ":( ("  
                        tmpExpr += "("                                             
                        tmpExpr += tmp
                        tmpExpr += ")"
                        if (stackModel.qsize() > 0):
                            tmpExpr += "+" #getGeneExp(config, gene)
                            ContT += 1
                            lenIndiv += 1

                    #end if
                #end while
                #tmpExpr= str(ContT)
                ContT += 2
                lenIndiv += 2
                Expr = str(ContT) + ":((" + tmpExpr + ")/" + str(NumExpr) + ")"
                #tmpExpr += ":( ("
                stackModel.put(Expr)
                #end if
            # End if

        # ********* Es un operador de desviacion estandard  ************/
        elif ((gene == OP_SDV)) :
            #print("Encontro OP_SDV")
            if (not stackModel.empty()) :
                tmpExpr = ""
                Expr = ""
                ContT = 0
                NumExpr = stackModel.qsize()

                while(stackModel.qsize() > 0 ) :
                    #print("Encontro expresion: ")
                    tmp = stackModel.get() # Se obtiene la ultima expresion de la pila de expresiones
                    #print(tmp)
                    strCont = tmp[0 : tmp.find(":")] # obtener el numero de elementos 
                    if (strCont.isnumeric()) :
                        cont1 = int(strCont)
                        ContT += cont1
                        tmp  = tmp[tmp.find(":") + 1 : len(tmp)] # Obtenemos solo la expresion
                        #tmpExpr= str(cont1 + 1)
                        #tmpExpr += ":( ("  
                        tmpExpr += "("                                             
                        tmpExpr += tmp
                        tmpExpr += ")"
                        if (stackModel.qsize() > 0):
                            tmpExpr += "+" #getGeneExp(config, gene)
                            ContT += 1
                            lenIndiv += 1

                    #end if
                #end while
                #tmpExpr= str(ContT)
                ContT += 2
                lenIndiv += 2
                Expr = "((" + tmpExpr + ")/" + str(NumExpr) + ")"
                #tmpExpr += ":( ("
                Expr = str(ContT) + ":((" + Expr + "))"
                stackModel.put(Expr)
                #end if
            # End if

        # ********* Es un operador IFG o IFL o IFE ************/
        elif ((gene ==  OP_IFG) or (gene == OP_IFL) or (gene == OP_IFE)) :
            if (not stackModel.empty()) :
               tmpExpr = ""   

        elif (gene == OP_NOOP) :  # Es NoOP, no hacemos nada
            if (not stackModel.empty()) :
                g1 = 0
            # End if
        else :
            tmpExpr = "1:"
            tmpExpr += "("+str(geneExpr)+")"
            stackModel.put(tmpExpr)
        # End if
    # End for (individuals)

    stackLen = stackModel.qsize()

    stackExpr = []
    if (not stackModel.empty()) :
        for j in range(stackLen):
            tmpExpr = str(lenIndiv)
            tmpExpr += ":"
            tmpExpr += str(stackLen)
            tmpExpr += ":"
            tmpExpr += stackModel.get()
            #stackExpr.insert(0,tmpExpr) 
            stackExpr.append(tmpExpr)
    else :
        tmpExpr = str(lenIndiv)
        tmpExpr += ":"
        tmpExpr += str(stackLen)
        tmpExpr += ":" 
        stackExpr.append(tmpExpr)      
    #end if
 
    #IndivLen:StackLen:ModelLen:ModelExpr
    return stackExpr



def m4gpModel(config, Model, Coef, Intercep) :
    lenIndiv = 0
    stackModel = LifoQueue()
    m4gpModel = []
    Expr = ""
    tmpExpr = ""

    var_ini = math.fabs(VAR_INI)
    maxVar = float((var_ini + config.nvar -1) * (-1))

    #print("maxvar:", maxVar)
    lenModel = len(Model)
    #print("getModelExpr. nvar:", config.nvar," Genes:",config.GenesIndividuals)
    for i in range(lenModel):
        gene = Model[i]
        if (gene == -11111) :
            break

        #geneExpr = getGeneExp(config, gene)
        #print("GeneExpr (", i, "): ", gene, " - ", geneExpr)
        lenIndiv += 1

        # ********************************* Es una constante ************************************/
        if ((gene >= MIN_CONSTANT) and (gene <= MAX_CONSTANT)) : # Es una constante
            stackModel.put(gene)

        # ********************************* Es una variable ************************************/
        elif ((gene >= maxVar) and (gene <= VAR_INI)) :  # Es una variable
            stackModel.put(gene)

        # ************ Es un operador de Suma,Resta,Division o Multiplicacion ******************/
        elif ((gene == OP_ADD) or (gene == OP_SUB) or (gene == OP_DIV) or (gene == OP_MUL)) :
            # Es Suma,Resta,Division o Multiplicacion
            tmpArr = []
            if (not stackModel.empty()) :
                tmp = stackModel.get() #Obtenemos el ultimo elemento del stack
                
                if (not stackModel.empty()) :
                    tmp2 = stackModel.get()
                    
                    tmpArr.append(tmp2)
                    tmpArr.append(tmp)
                    tmpArr.append(gene)

                    stackModel.put(tmpArr)
                else :
                    stackModel.put(tmp)
                # End if
            # End if

        # ********* Es un operador de seno, coseno, exponente, logaritmo, absoluto, tangente, tangente hyperbolica ************/
        elif ((gene == OP_SIN) or (gene == OP_COS) or (gene == OP_EXP) or (gene == OP_LOG) or (gene == OP_ABS) or (gene == OP_TAN) or (gene == OP_TANH) or (gene == OP_SQRT)) :
            tmpArr = []
            if (not stackModel.empty()) :
                tmp = stackModel.get()
                tmpArr.append(tmp)
                tmpArr.append(gene)
                stackModel.put(tmpArr)
            # End if

          # ********* Es un Sumatoria, Producto ************/
        elif ((gene == OP_SUM) or (gene == OP_PRD) ) :
            tmpArr = []
            if (not stackModel.empty()) :
                while(stackModel.qsize() > 0 ) :
                    #print("M4gp Encontro expresion: ")
                    tmp = stackModel.get() # Se obtiene la ultima expresion de la pila de expresiones
                    #print(tmp)
                
                    tmpArr.append(tmp)

                    if (stackModel.qsize() > 0 and gene == OP_SUM ) :
                        tmpArr.append(OP_ADD)
                    if (stackModel.qsize() > 0 and gene == OP_PRD ) :
                        tmpArr.append(OP_MUL)
                #end while

                stackModel.put(tmpArr)
            #end if

          # ********* Promedio ************/
        elif (gene == OP_AVG) :
            tmpArr = []
            if (not stackModel.empty()) :
                nSize = stackModel.qsize()
                while(stackModel.qsize() > 0 ) :
                    #print("M4gp AVG Encontro expresion: ")
                    tmp = stackModel.get() # Se obtiene la ultima expresion de la pila de expresiones
                    #print(tmp)
                
                    tmpArr.append(tmp)
                    if (stackModel.qsize() > 0 ) :
                        tmpArr.append(OP_ADD)
                #end while
                tmpArr.append(nSize)
                tmpArr.append(OP_DIV)
                
                stackModel.put(tmpArr)
            #end if

          # ********* Promedio, Desv Standard ************/
        elif (gene == OP_SDV) :
            tmpArr = []
            if (not stackModel.empty()) :
                nSize = stackModel.qsize()
                while(stackModel.qsize() > 0 ) :
                    #print("M4gp SDV Encontro expresion: ")
                    tmp = stackModel.get() # Se obtiene la ultima expresion de la pila de expresiones
                    #print(tmp)
                
                    tmpArr.append(tmp)
                    if (stackModel.qsize() > 0 ) :
                        tmpArr.append(OP_ADD)
                #end while
                tmpArr.append(nSize)
                tmpArr.append(OP_DIV)


                stackModel.put(tmpArr)
            #end if
                                  
        elif (gene == OP_NOOP) :  # Es NoOP, no hacemos nada
            if (not stackModel.empty()) :
                g1 = g1
            # End if
        else :
            g1 = gene
        # End if
    # End for (individuals)

    stackLen = stackModel.qsize()
  
    #IndivLen:StackLen:ModelLen:ModelExpr
    return stackModel

def m4gpBuildExpr(tmp1, nvoModel) :
    if isinstance(tmp1, list):
        lenTmp = len(tmp1)
        for k in range(lenTmp):
            tmp4 = tmp1[lenTmp-k-1]
            nvoModel = m4gpBuildExpr(tmp4, nvoModel)
        nvoModel.append(OP_ADD)
    # End for
    else :    
        nvoModel.insert(0,tmp1)
    return nvoModel

