# *********************************************************************
# Name: m5gpMod1.py
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

# *************************** Initialize population ******************************** 
def initialize_population (
        numIndividuals,
        nvar,
        sizeMaxDepthIndividual,
        maxRandomConstant,
        genOperatorProb,
        genVariableProb,
        genConstantProb,
        genNoopProb,
        useOpIF,
        hOperators,
        h_cdf ) :
    

    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    gridsize = MaxOcup["GridSize"]
    blocksize = MaxOcup["BlockSize"]
    
    # Initialize a state for each thread
    tiempo = int(repr(int((time.time() % 1)*1000000000))[-6:])
    cu_states = create_xoroshiro128p_states(blocksize*gridsize, seed=tiempo)

    #New code
    # stream_h2d = cuda.stream()
    # stream_d2h = cuda.stream()

    # # Host buffer "pinned"
    # hInitialPopulation = cuda.pinned_array((gpG.sizePopulation, ), dtype=np.int32)
    # dInitialPopulation = cuda.device_array((gpG.sizePopulation, ), dtype=np.int32)

    # # Copia H→D asíncrona
    # hInitialPopulation[:] = np.zeros(gpG.sizePopulation).astype(np.int32)
    # dInitialPopulation.copy_to_device(hInitialPopulation, stream=stream_h2d)

    #Old code
    hInitialPopulation = np.zeros((gpG.sizePopulation), dtype=np.float32) 
    dInitialPopulation = cuda.to_device(hInitialPopulation)
    dOperators = cuda.to_device(hOperators)
    d_cdf = cuda.to_device(h_cdf)

    start_time = time.time()
      
    # print( "initialize_population - Gridsize: ", gridsize, "Blocksize:", blocksize)
    gpCuda.initialize_population[gridsize, blocksize](cu_states,
                                        dInitialPopulation,
                                        numIndividuals,
                                        nvar,
                                        sizeMaxDepthIndividual,
                                        maxRandomConstant,
                                        genOperatorProb,
                                        genVariableProb,
                                        genConstantProb,
                                        genNoopProb,
                                        useOpIF, 
                                        dOperators,
                                        d_cdf ) 
    elapsed = time.time() - start_time

    #cuda.synchronize()
    hInitialPopulation = dInitialPopulation.copy_to_host()

    # Copia D→H asíncrona (cuando no haya dependencia con stream_h2d, usa otro stream)
    ##dInitialPopulation.copy_to_host(hInitialPopulation, stream=stream_d2h)
    ##stream_d2h.synchronize()

    Ops = (numIndividuals * sizeMaxDepthIndividual)
    #print("InitialPopulation elapsed_time: " + str(elapsed))
    gpG.WriteCSV_OpS("InitialPopulation", elapsed,Ops,True)
 
    return hInitialPopulation
# ******************** -- End of Initialize population -- ***********************


# ***************************  Compute Individuals  ****************************
def compute_individuals(
        hInitialPopulation,
        hData,
        numIndividuals,
        GenesIndividuals,
        nrowTrain,
        nvar,
        getStackModel ) :

  # Total elements of the data train matrix to form
  totalElements = nrowTrain * nvar

  # Total elements of the number of individuals in the initial population
  sizeIndividuals = numIndividuals 
  
  # Total elements of semantics elements for the entire population with training data
  sizeIndividualsTrain = numIndividuals * nrowTrain 

  # Total elements of the training data
  sizeDataTrain = totalElements

  # Total elements of the resulting Model
  sizeModel = GenesIndividuals * numIndividuals * nrowTrain

  # Total elements of Population
  sizePopulation = numIndividuals * GenesIndividuals

  # Total elements of Stack (size Stack)
  sizeStack = sizePopulation * nrowTrain

  # Total elements of Idx Stack
  sizeStackIdx = sizeIndividuals  * nrowTrain

  # Calculate the available memory for slide individuals blocks 
  memRequired = (np.dtype(np.float32).itemsize) * (sizePopulation + 
                          sizeIndividualsTrain + 
                          sizeDataTrain + 
                          sizeStack + 
                          sizeStackIdx + 
                          sizeModel)


  memRest = gpG.free_mem-memRequired
  memUsePercent = memRequired/gpG.free_mem
  memUsePercent2 = memUsePercent - math.floor(memUsePercent)
  
  #print("memRequired: ",memRequired, "memFree: ", gpG.free_mem)
  #print("1. memUsePercent: ", memUsePercent, " memUsePercent2: ", memUsePercent2, )

  memUsePercent = math.ceil(memUsePercent)
  #print("2. memUsePercent: ", memUsePercent, " memUsePercent2: ", memUsePercent2, )

  if (memUsePercent2 > 0.85):
    memUsePercent = memUsePercent + 1
  
  if (memUsePercent <= 1) :
    memUsePercent = 1

  #print("3. memUsePercent: ", memUsePercent, " memUsePercent2: ", memUsePercent2, )

  # We obtain the number of blocks that are necessary to evaluate the population 
  # with respect to the data. The goal is to use no more than 85% of the available 
  # memory on the GPU device when processing each block.      
  numIndividualsBlock = math.ceil(numIndividuals / memUsePercent)

  initialBlock = 0
  finalBlock = numIndividualsBlock 

  hData = np.reshape(hData, -1)
  dDataTrain = cuda.to_device(hData)

  # ******************  Individuals Evaluation  ********************
  # Invokes the GPU to interpret the initial population with data train
  # divide initial population in blocks to fit in available memory 

  hOutIndividuals = [] 
  hOutIndividualsBlock = []
  # print("sizeStack:", sizeStack)
  hStack = np.zeros((sizeStack), dtype=np.float32)
  hStackIdx = np.zeros((sizeStackIdx), dtype=np.float32)
  hStackModel = []
  if (getStackModel == 1):
    hStackModel = np.zeros((sizeModel), dtype=np.float32)
  
  dOutIndividualsBlock = 0
  pBlock1 = 0
  pBlocki_ant = 0
  pBlocks_ant = 0

  start_time = time.time()
  Ops = 0

  elapsed1 = time.time() - start_time
  #print("compute_individuals 1 (" + str(pBlock1) + ")", elapsed1, Ops)
  #gpG.WriteCSV_OpS("compute_individuals 1 ", elapsed1,Ops)

  # If necessary, due to the amount of memory required, 
  # the population to be evaluated is divided into blocks 
  # so as not to saturate the memory..
  while(finalBlock <= numIndividuals) :  
    sizePopulationBlock = numIndividualsBlock * GenesIndividuals
    sizeIndividualsBlock = numIndividualsBlock * nrowTrain
    memStackBlock = sizePopulationBlock * nrowTrain
    memStackIdxBlock = sizeIndividualsBlock
    sizeModelBlock = numIndividualsBlock * GenesIndividuals * nrowTrain
    totalSemanticElementsBlock = numIndividualsBlock * nrowTrain
    #sizeIndividualsBlock = numIndividualsBlock * nrowTrain

    hStackBlock = np.zeros((memStackBlock), dtype=np.float32)
    hStackIdxBlock = np.zeros((memStackIdxBlock), dtype=np.float32)
    
    #hStackModelBlock = np.zeros((sizeModelBlock), dtype=np.float32)
    if getStackModel == 1:
      hStackModelBlock = np.zeros((sizeModelBlock), dtype=np.float32)
    else:
      hStackModelBlock = np.zeros((1,), dtype=np.float32)  # Dummy pequeño por si el kernel espera algo
        
    hOutIndividualsBlock = np.zeros((sizeIndividualsBlock), dtype=np.float32)
    hArrayTmp = np.zeros((numIndividuals), dtype=np.float32)    

    # Get initial population block for evaluate individuals
    if (finalBlock ==  numIndividuals and pBlock1 == 0):
      hInitialPopulationBlock = hInitialPopulation
    else:
      hInitialPopulationBlock = hInitialPopulation[(initialBlock*GenesIndividuals):(finalBlock*GenesIndividuals)]
       
    dInitialPopulationBlock = cuda.to_device(hInitialPopulationBlock)
    dOutIndividualsBlock = cuda.to_device(hOutIndividualsBlock)   
    dStackBlock = cuda.to_device(hStackBlock)
    dStackIdxBlock = cuda.to_device(hStackIdxBlock)
    dStackModelBlock = cuda.to_device(hStackModelBlock)
    dArrayTmp = cuda.to_device(hArrayTmp)
        
    # MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElementsBlock)
    # blocksize = MaxOcup["BlockSize"]
    # gridsize = MaxOcup["GridSize"]    


      
    elapsed2 = time.time() - start_time - elapsed1
    #print("compute_individuals 2 (" + str(pBlock1) + ")", elapsed2,Ops)
    #gpG.WriteCSV_OpS("compute_individuals 2 ", elapsed2,Ops)
    
    #print("numIndividuals: ", numIndividualsBlock, " ", "totalSemanticElementsBlock:", totalSemanticElementsBlock)
    if (numIndividualsBlock == 1) :
      MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElementsBlock, 1)
      #print("numIndividuals: ", numIndividualsBlock)
    else:
      MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElementsBlock)
    
    blocks_per_grid = MaxOcup["GridSize"]
    threads_per_block = MaxOcup["BlockSize"]
    # print( "compute_individuals - Gridsize: ", blocks_per_grid, "Blocksize:", threads_per_block)
    
    #print("compute_individuals ("+ str(totalSemanticElementsBlock) + ") - threads_per_block: " + str(threads_per_block) + " blocks_per_grid: " + str(blocks_per_grid))
    gpCuda.compute_individuals[blocks_per_grid, threads_per_block](
                        dInitialPopulationBlock,
                        dOutIndividualsBlock,
                        dDataTrain,
                        numIndividualsBlock,
                        GenesIndividuals,
                        nrowTrain,
                        nvar,
                        dStackBlock,
                        dStackIdxBlock,
                        getStackModel,
                        dStackModelBlock,
                        dArrayTmp
    )
    elapsed3 = time.time() - start_time - elapsed2 - elapsed1
    #print("compute_individuals 3 (" + str(pBlock1) + ")", elapsed3,Ops)
    #gpG.WriteCSV_OpS("compute_individuals 3 ", elapsed3,Ops)

    cuda.synchronize()
    
    elapsed4 = time.time() - start_time - elapsed3 - elapsed2 - elapsed1
    #gpG.WriteCSV_OpS("compute_individuals 4 ", elapsed4,Ops)

    # Return blocks from Device to host
    hOutIndividualsBlock = dOutIndividualsBlock.copy_to_host()
    hStackBlock = dStackBlock.copy_to_host()
    hStackIdxBlock = dStackIdxBlock.copy_to_host()

    elapsed5 = time.time() - start_time - elapsed4 - elapsed3 - elapsed2 - elapsed1
    # print("compute_individuals 5 (" + str(pBlock1) + ")", elapsed5,Ops)
    #gpG.WriteCSV_OpS("compute_individuals 5 ", elapsed5,Ops)

    if (finalBlock >= numIndividuals and pBlock1 == 0) :
      hOutIndividuals = hOutIndividualsBlock
      hStackIdx = hStackIdxBlock
      hStack = hStackBlock
    else :
      # Join device blocks with in one local block 
      hOutIndividuals = np.hstack((hOutIndividuals, hOutIndividualsBlock))
      pBlocki = pBlocki_ant + hStackIdxBlock.shape[0]
      hStackIdx[pBlocki_ant:pBlocki] = hStackIdxBlock  
      pBlocki_ant = pBlocki

      pBlocks = pBlocks_ant + hStackBlock.shape[0]

      # print("hstack.shape[0]:", hStack.shape[0], "pBlocks_ant: ", pBlocks_ant, " pBlocks:", pBlocks, " hStackBlock.shape[0]:",  hStackBlock.shape[0])
      hStack[pBlocks_ant:pBlocks] = hStackBlock
      pBlocks_ant = pBlocks

    #end if

    pBlock1 = pBlock1 + 1

    elapsed6 = time.time() - start_time - elapsed5 - elapsed4 - elapsed3 - elapsed2 - elapsed1
    # print("compute_individuals 6 (" + str(pBlock1) + ")", elapsed6,Ops)
    #gpG.WriteCSV_OpS("compute_individuals 6 ", elapsed6,Ops)
        
    if (finalBlock >= numIndividuals) :
      break

    initialBlock = finalBlock 
    finalBlock = initialBlock + numIndividualsBlock
    if (finalBlock > numIndividuals) :
      numIndividualsBlock = numIndividuals - initialBlock
      finalBlock = numIndividuals
  # End while

  elapsed7 = time.time() - start_time - elapsed6 - elapsed5 - elapsed4 - elapsed3 - elapsed2 - elapsed1
  # print("compute_individuals 7 (" + str(pBlock1) + ")", elapsed7,Ops)

  elapsed = time.time() - start_time 
  Ops = (numIndividuals  * nrowTrain * GenesIndividuals)
  #print("compute_individuals (" + str(pBlock1) + ")", elapsed,Ops)
  # gpG.WriteCSV_OpS("compute_individuals (" + str(pBlock1) + ")", elapsed,Ops)

  del hStackModelBlock
  #Free local memory 
  del hStackBlock
  del hStackIdxBlock
  del hInitialPopulationBlock
  del hOutIndividualsBlock 
  
  #Free gpu vectors memory 
  del dStackBlock
  del dStackIdxBlock
  del dStackModelBlock
  del dOutIndividualsBlock
  del dInitialPopulationBlock
  gc.collect()

  return hOutIndividuals, hStack, hStackIdx, hStackModel
# *************************  End of Compute Individuals  **************************



# ********************************************************************************
def compute_individuals2(
        hInitialPopulation,
        hData,
        numIndividuals,
        GenesIndividuals,
        nrowTrain,
        nvar,
        getStackModel):

    # Total elements of the data train matrix to form
    totalElements = nrowTrain * nvar

    # Total elements of the number of individuals in the initial population
    sizeIndividuals = numIndividuals
    
    # Total elements of semantics elements for the entire population with training data
    sizeIndividualsTrain = numIndividuals * nrowTrain

    # Total elements of the training data
    sizeDataTrain = totalElements

    # Total elements of the resulting Model
    sizeModel = GenesIndividuals * numIndividuals * nrowTrain

    # Total elements of Population
    sizePopulation = numIndividuals * GenesIndividuals

    # Total elements of Stack (size Stack)
    sizeStack = sizePopulation * nrowTrain

    # Total elements of Idx Stack
    sizeStackIdx = sizeIndividuals * nrowTrain

    # ********** Cálculo de memoria requerida (GPU) **********
    memRequired = (np.dtype(np.float32).itemsize) * (
        sizePopulation +
        sizeIndividualsTrain +
        sizeDataTrain +
        sizeStack +
        sizeStackIdx +
        sizeModel
    )

    memRest = gpG.free_mem - memRequired
    memUsePercent = memRequired / gpG.free_mem
    memUsePercent2 = memUsePercent - math.floor(memUsePercent)

    memUsePercent = math.ceil(memUsePercent)

    if memUsePercent2 > 0.85:
        memUsePercent = memUsePercent + 1

    if memUsePercent <= 1:
        memUsePercent = 1

    # We obtain the number of blocks that are necessary to evaluate the population 
    # with respect to the data. The goal is to use no more than 85% of the available 
    # memory on the GPU device when processing each block.      
    numIndividualsBlock = math.ceil(numIndividuals / memUsePercent)

    initialBlock = 0
    finalBlock = numIndividualsBlock

    # Aplanamos datos de entrenamiento y los mandamos a GPU
    hData = np.reshape(hData, -1)
    dDataTrain = cuda.to_device(hData)

    # ******************  Individuals Evaluation  ********************

    # 🔹 PREASIGNAR vectores grandes EN VEZ DE USAR np.hstack
    hOutIndividuals = np.zeros(sizeIndividualsTrain, dtype=np.float32)
    hStack = np.zeros(sizeStack, dtype=np.float32)
    hStackIdx = np.zeros(sizeStackIdx, dtype=np.float32)

    if getStackModel == 1:
        hStackModel = np.zeros(sizeModel, dtype=np.float32)
    else:
        hStackModel = []

    dOutIndividualsBlock = 0

    start_time = time.time()
    Ops = 0

    pBlock1 = 0  # solo para contar bloques, ya no se usa para lógica de copia

    while finalBlock <= numIndividuals:
        # Tamaños por bloque (en individuos)
        numIndividualsBlockLocal = finalBlock - initialBlock  # OJO: puede ser menor en el último bloque

        sizePopulationBlock = numIndividualsBlockLocal * GenesIndividuals
        sizeIndividualsBlock = numIndividualsBlockLocal * nrowTrain
        memStackBlock = sizePopulationBlock * nrowTrain
        memStackIdxBlock = sizeIndividualsBlock
        sizeModelBlock = numIndividualsBlockLocal * GenesIndividuals * nrowTrain
        totalSemanticElementsBlock = numIndividualsBlockLocal * nrowTrain

        # ********** Arrays de host POR BLOQUE **********
        hStackBlock = np.zeros(memStackBlock, dtype=np.float32)
        hStackIdxBlock = np.zeros(memStackIdxBlock, dtype=np.float32)
        hOutIndividualsBlock = np.zeros(sizeIndividualsBlock, dtype=np.float32)
        hArrayTmp = np.zeros(numIndividualsBlockLocal, dtype=np.float32)  # si el kernel lo permite

        if getStackModel == 1:
            hStackModelBlock = np.zeros(sizeModelBlock, dtype=np.float32)
        else:
            # Dummy pequeño por si el kernel espera algo
            hStackModelBlock = np.zeros(1, dtype=np.float32)

        # ********** Población inicial por bloque **********
        if finalBlock == numIndividuals and pBlock1 == 0:
            # Primer y único bloque (no se fragmentó la población)
            hInitialPopulationBlock = hInitialPopulation
        else:
            startPop = initialBlock * GenesIndividuals
            endPop = finalBlock * GenesIndividuals
            hInitialPopulationBlock = hInitialPopulation[startPop:endPop]

        # ********** Copiar a GPU **********
        dInitialPopulationBlock = cuda.to_device(hInitialPopulationBlock)
        dOutIndividualsBlock = cuda.to_device(hOutIndividualsBlock)
        dStackBlock = cuda.to_device(hStackBlock)
        dStackIdxBlock = cuda.to_device(hStackIdxBlock)
        dArrayTmp = cuda.to_device(hArrayTmp)

        if getStackModel == 1:
            dStackModelBlock = cuda.to_device(hStackModelBlock)
        else:
            # Dummy pequeño por si el kernel lo requiere como parámetro
            dStackModelBlock = cuda.to_device(np.zeros(1, dtype=np.float32))

        # MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElementsBlock)
        # blocksize = MaxOcup["BlockSize"]
        # gridsize = MaxOcup["GridSize"]

        MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElementsBlock)
        blocks_per_grid = MaxOcup["GridSize"]
        threads_per_block = MaxOcup["BlockSize"]
        
        # ********** Llamada al kernel en GPU **********
        gpCuda.compute_individuals[blocks_per_grid, threads_per_block](
            dInitialPopulationBlock,
            dOutIndividualsBlock,
            dDataTrain,
            numIndividualsBlockLocal,
            GenesIndividuals,
            nrowTrain,
            nvar,
            dStackBlock,
            dStackIdxBlock,
            getStackModel,
            dStackModelBlock,
            dArrayTmp
        )

        cuda.synchronize()

        # ********** Copiar resultados DE GPU A HOST **********
        hOutIndividualsBlock = dOutIndividualsBlock.copy_to_host()
        hStackBlock = dStackBlock.copy_to_host()
        hStackIdxBlock = dStackIdxBlock.copy_to_host()
        if getStackModel == 1:
            hStackModelBlock = dStackModelBlock.copy_to_host()

        # ********** COPIA POR SLICES EN VEZ DE np.hstack **********

        # 1) hOutIndividuals: tamaño total = numIndividuals * nrowTrain
        #    Cada individuo ocupa nrowTrain posiciones.
        out_start = initialBlock * nrowTrain
        out_end = out_start + sizeIndividualsBlock
        hOutIndividuals[out_start:out_end] = hOutIndividualsBlock

        # 2) hStackIdx: misma lógica que hOutIndividuals
        idx_start = initialBlock * nrowTrain
        idx_end = idx_start + sizeIndividualsBlock
        hStackIdx[idx_start:idx_end] = hStackIdxBlock

        # 3) hStack y hStackModel:
        #    Flatten por genes e individuos:
        #    población global = numIndividuals * GenesIndividuals
        #    en este bloque: numIndividualsBlockLocal * GenesIndividuals
        pop_start = initialBlock * GenesIndividuals
        pop_end = pop_start + sizePopulationBlock

        stack_start = pop_start * nrowTrain
        stack_end = stack_start + memStackBlock

        hStack[stack_start:stack_end] = hStackBlock

        if getStackModel == 1:
            hStackModel[stack_start:stack_end] = hStackModelBlock

        # ********** Limpiar temporales de GPU para este bloque **********
        del dInitialPopulationBlock
        del dOutIndividualsBlock
        del dStackBlock
        del dStackIdxBlock
        del dArrayTmp
        del dStackModelBlock
        gc.collect()

        pBlock1 += 1

        # ********** Actualizar rangos de bloque **********
        if finalBlock >= numIndividuals:
            break

        initialBlock = finalBlock
        finalBlock = initialBlock + numIndividualsBlock
        if finalBlock > numIndividuals:
            finalBlock = numIndividuals

    # ********** Fin del while **********

    elapsed = time.time() - start_time
    Ops = numIndividuals * nrowTrain * GenesIndividuals
    # print("compute_individuals (" + str(pBlock1) + ")", elapsed, Ops)
    # gpG.WriteCSV_OpS("compute_individuals (" + str(pBlock1) + ")", elapsed, Ops)

    # Limpieza final de algunos temporales de host
    del hStackBlock
    del hStackIdxBlock
    del hInitialPopulationBlock
    del hOutIndividualsBlock
    if getStackModel == 1:
        del hStackModelBlock
    gc.collect()

    return hOutIndividuals, hStack, hStackIdx, hStackModel
# *************************  End of Compute Individuals 2  **************************



# ****************************  Evaluate Individuals  *****************************
def ComputeError(self,
                hOutIndividuals, 
                hDataY, 
                numIndividuals, 
                nrowTrain,
                hStack, 
                hStackIdx,
                evaluationMethod) :
   
    coefArr_p = []
    intercepArr_p = []    
    cuModel_p = []

    start_time = time.time()

    result_train_p = 0
    hFit = np.zeros((gpG.sizeIndividuals), dtype=np.float32)
    dFit = cuda.to_device(hFit)

    dOutIndividuals = cuda.to_device(hOutIndividuals)
    dDataY = cuda.to_device(hDataY)

    #gridsize = gpCuda.gpuMaxUseProc(self.Individuals, blocksize)
    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    gridsize = MaxOcup["GridSize"]
    blocksize = MaxOcup["BlockSize"]

    #print( "Gridsize: ", gridsize, "Blocksize:", blocksize)                                               
    if evaluationMethod == 0 :  #0=RMSE
        #print("RMSE")
        gpCuda.computeRMSE[gridsize, blocksize ](
                        dOutIndividuals, 
                        dDataY, 
                        dFit, 
                        numIndividuals, 
                        nrowTrain ) 

        hFit = dFit.copy_to_host()
        # This section makes use of the isamin of cublas function to determine
        # the position of the best individual in initial Population using RMSE
        result_off = gpG.np.argmin(hFit)        
        indexBestOffspring = result_off

        result_w = gpG.np.argmax(hFit)	
        indexWorstOffspring = result_w       

        
    elif evaluationMethod == 1 :  #1=R2 :
        gpCuda.computeR2[gridsize, blocksize](
                        dOutIndividuals, 
                        dDataY, 
                        dFit, 
                        numIndividuals, 
                        nrowTrain )       

        hFit = dFit.copy_to_host()

        # This section makes use of the isamax of cublas function to determine
        # the position of the best individual in initial Population using R2
        # #make a handle to the function of tf.cublas 
        result_off = gpG.np.argmax(hFit)        
        indexBestOffspring = result_off

        result_w = gpG.np.argmin(hFit)	
        indexWorstOffspring = result_w      

    elif (evaluationMethod == 2 or #M4GP - 2=cuML LinearRegression
        evaluationMethod == 3 or #M4GP - 3=cuML Lasso regularization
        evaluationMethod == 4 or #M4GP - 4=cuML Ridge regularization
        evaluationMethod == 5 or #M4GP - 5=cuML kernel Ridge Regression
        evaluationMethod == 6 or #M4GP - 6=cuML Elasticnet regularization 
        evaluationMethod == 7 or #M4GP - 7=cuML MiniBatch none regularization
        evaluationMethod == 8 or #M4GP - 8=cuML MiniBatch lasso regularization
        evaluationMethod == 9 or #M4GP - 9=cuML MiniBatch ridge regularization
        evaluationMethod == 10) : #M4GP - 10=cuML MiniBatch elasticnet regularization

        #start_time = time.time()
        coefArr = []
        intercepArr = []
        cuModel = []

        hFit, cuModel, coefArr, intercepArr = gpCuM.EvaluateCuml2(self, hStack, hStackIdx, hFit, hDataY)

        #elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        #print(f"Time cuML lapsed: {elapsed}")
  
        #dFit = cuda.to_device(hFit)
        if (self.scorer==0) or (self.scorer==1):
          result_off = gpG.np.argmin(hFit)        
          indexBestOffspring = result_off

          result_w = gpG.np.argmax(hFit)	
          indexWorstOffspring = result_w   
        elif  (self.scorer==2) :
          result_off = gpG.np.argmax(hFit)        
          indexBestOffspring = result_off

          result_w = gpG.np.argmin(hFit)	
          indexWorstOffspring = result_w       

        # Obtenemos los coeficientes y el modelo del 
        # mejor individuo generados por cuML
        coefArr_p = coefArr[indexBestOffspring]
        intercepArr_p = intercepArr[indexBestOffspring]  
        cuModel_p = cuModel[indexBestOffspring]
     
    # end if (Evaluation methods)

    elapsed = time.time() - start_time
    Ops = (numIndividuals * nrowTrain)
    #print("compute_error", elapsed, Ops)
    #gpG.WriteCSV_OpS("compute_error", elapsed,Ops)    
 
    return hFit, indexBestOffspring,  indexWorstOffspring, coefArr_p, intercepArr_p, cuModel_p
# *************************  End of Evaluate Individuals  **************************

# *******************************  Select Tournament  ******************************
def select_tournament(
                    hInitialPopulation,
                    hFit,
                    numIndividuals,
                    GenesIndividuals ) :
    

    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    gridsize = MaxOcup["GridSize"]
    blocksize = MaxOcup["BlockSize"]
    
    tiempo = int(repr(int((time.time() % 1)*1000000000))[-6:])
    # Initialize a state for each thread
    cu_states = create_xoroshiro128p_states(blocksize*gridsize, seed=tiempo)
       

    hNewPopulation  = np.zeros((gpG.sizePopulation), dtype=np.float32) 
    hBestParentsTournament = np.zeros((gpG.sizeIndividuals), dtype=np.int32)

    dBestParentsTournament = cuda.to_device(hBestParentsTournament)
    dInitialPopulation = cuda.to_device(hInitialPopulation)
    dNewPopulation = cuda.to_device(hNewPopulation)

    dFit = cuda.to_device(hFit)

    start_time = time.time()
      
    gpCuda.parent_select_tournament[gridsize, blocksize](cu_states,
                              dNewPopulation,
                              dInitialPopulation,
                              dFit,
                              dBestParentsTournament,
                              gpG.sizeTournament,
                              numIndividuals,
                              GenesIndividuals   )

    hNewPopulation = dNewPopulation.copy_to_host()
    hBestParentsTournament = dBestParentsTournament.copy_to_host()   

    elapsed = time.time() - start_time
    Ops = (numIndividuals * gpG.sizeTournament)
    #print("tournament("+str(gpG.sizeTournament)+")", elapsed,Ops)
    gpG.WriteCSV_OpS("tournament("+str(gpG.sizeTournament)+")", elapsed,Ops)
   
    return hNewPopulation, hBestParentsTournament
# ****************************  End of Select Tournament  ****************************

# *********************************  UMAD Mutation  **********************************
def umadMutation(self,
                 hInitialPopulation,
                 hBestParentsTournament,
                 numIndividuals,
                 h_cdf) :

    MaxOcup = gpCuda.gpuMaxUseProc(numIndividuals)
    gridsize = MaxOcup["GridSize"] 
    blocksize = MaxOcup["BlockSize"]
    
    tiempo = int(repr(int((time.time() % 1)*1000000000))[-6:])
    # Initialize a state for each thread
    cu_states = create_xoroshiro128p_states(blocksize*gridsize, seed=tiempo)

    hNewPopulation  = np.zeros((gpG.sizePopulation), dtype=np.float32) 
    dNewPopulation = cuda.to_device(hNewPopulation)
    dInitialPopulation = cuda.to_device(hInitialPopulation)
    dBestParentsTournament = cuda.to_device(hBestParentsTournament)
    dOperators = cuda.to_device(self.valid_functions_set)
    d_cdf = cuda.to_device(h_cdf)

    start_time = time.time()
       
    gpCuda.umadMutation[gridsize, blocksize](cu_states,
                        dNewPopulation,
                        dInitialPopulation,
                        dBestParentsTournament,
                        numIndividuals,
                        self.GenesIndividuals,
                        self.nrowTrain,
                        self.nvar,
                        self.mutationProb,
                        self.mutationDeleteRateProb,
                        self.maxRandomConstant,
                        self.genOperatorProb,
                        self.genVariableProb,
                        self.genConstantProb,
                        self.genNoopProb,
                        self.useOpIF,
                        dOperators,
                        d_cdf)

    hNewPopulation = dNewPopulation.copy_to_host()

    elapsed = time.time() - start_time
    Ops = (numIndividuals * self.GenesIndividuals) 
    #print("umadMutation", elapsed,Ops)
    gpG.WriteCSV_OpS("umadMutation", elapsed,Ops) 

    return hNewPopulation
# ****************************  End of UMAD Mutation  ******************************

# *****************************  Survival (Elitist)  *******************************
def Survival(self,
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
            stackBestModelNew) :
    
    idx_a1 = indexWorstOffspring * self.GenesIndividuals
    idx_b1 = indexWorstOffspring * self.GenesIndividuals + self.GenesIndividuals
    idx_a2 = indexBestIndividual_p * self.GenesIndividuals
    idx_b2 = indexBestIndividual_p * self.GenesIndividuals + self.GenesIndividuals

    if (self.evaluationMethod == 0 or self.evaluationMethod >= 2) and  (self.scorer !=2) :
        
        # Checamos si la nueva generacion es mejor que la anterior
        if (hFit[indexBestIndividual_p] < hFitNew[indexBestOffspring]) :
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

          # Si es un metodo decuML, copiamos los coeficientes del
          # mejor individuo de la nueva generacion como papa para
          # la siguiente generacion
          if (self.evaluationMethod >= 2) :
            coefArr_p = coefArrNew
            intercepArr_p = intercepArrNew 
            cuModel_p = cuModelNew
            stackBestModel_p = stackBestModelNew
    # End if
    elif (self.evaluationMethod == 1) or (self.scorer==2) :
        # Checamos si el mejor padre de la anterior poblacion es mejor que el mejor hijo en la nueva poblacion*/
        if (hFit[indexBestIndividual_p] > hFitNew[indexBestOffspring]) :
          # Pasa el mejor individuo de la anterior poblacion a la posicion del peor individuo dela nueva poblacion */
          hNewPopulation[idx_a1:idx_b1] = hInitialPopulation[idx_a2:idx_b2]
          # Ahora el peor hijo es el mejor padre
          hFitNew[indexWorstOffspring] = hFit[indexBestIndividual_p]
          indexBestIndividual_p = indexWorstOffspring
        else :
          indexBestIndividual_p = indexBestOffspring

          if (self.evaluationMethod >= 2) :
            coefArr_p = coefArrNew
            intercepArr_p = intercepArrNew 
            cuModel_p = cuModelNew
            stackBestModel_p = stackBestModelNew
                      
        # End if
    # End if
    return hNewPopulation, indexBestIndividual_p, coefArr_p, intercepArr_p, cuModel_p, stackBestModel_p
# ***************************  End of Survival (Elitist)  *****************************

     # ***********************    NEW REPLACE   ***********************
def replace(self,
              hInitialPopulation,
              hNewPopulation, 
              hFit,
              hFitNew) :
  
  dInitialPopulation = cuda.to_device(hInitialPopulation)
  dNewPopulation = cuda.to_device(hNewPopulation)
  dFit = cuda.to_device(hFit)
  dFitNew = cuda.to_device(hFitNew)  

  # Move new population to Initial population for individuals and Fits 
  MaxOcup = gpCuda.gpuMaxUseProc(self.Individuals)
  gridsize = MaxOcup["GridSize"]
  blocksize = MaxOcup["BlockSize"]
  
  gpCuda.replace[gridsize, blocksize](dInitialPopulation, 
          dNewPopulation, 
          dFit,
          dFitNew, 
          self.Individuals, 
          self.GenesIndividuals)

  # Copiamos valores del dispositivo GPU al  host (locales)
  hFit = dFit.copy_to_host()      
  hInitialPopulation = dInitialPopulation.copy_to_host()
       
  return hInitialPopulation, hFit
# *********************** END NEW REPLACE ***********************

def getStackBestModel(
        hModelPopulation,
        hData,
        numIndividuals,
        GenesIndividuals,
        nrowTrain,
        nvar) :

  numIndividuals = 1
  #nrowTrain = 1
  hData = np.reshape(hData, -1)
  #print("hData:")
  #print(hData)
  #GenesIndiv = hInitialPopulation.shape[0] # self.GenesIndividuals

  # Calculate memory por size vectors
  sizeIndividuals = numIndividuals * nrowTrain 
  sizePopulation = numIndividuals * GenesIndividuals
  memStack = sizePopulation * nrowTrain
  memStackIdx = sizeIndividuals
  sizeModel = GenesIndividuals * numIndividuals * nrowTrain
  totalSemanticElements = numIndividuals * nrowTrain

  #local vector
  hStack = []
  hStackIdx = []
  hStackModel = []
  hStack = np.zeros((memStack), dtype=np.float32)
  hStackIdx = np.zeros((memStackIdx), dtype=np.float32)
  hStackModel = np.zeros((sizeModel), dtype=np.float32)
  hOutIndividuals = np.zeros((sizeIndividuals), dtype=np.float32) 
  hArrayTmp = np.zeros((numIndividuals), dtype=np.float32) 

  # Copy vectors to gpu device
  dModelPopulation = cuda.to_device(hModelPopulation)
  dData = cuda.to_device(hData)
  dStack = cuda.to_device(hStack)
  dStackIdx = cuda.to_device(hStackIdx)
  dStackModel = cuda.to_device(hStackModel)
  dOutIndividuals = cuda.to_device(hOutIndividuals)
  dArrayTmp = cuda.to_device(hArrayTmp)   

  #print("hModelPopulation:", hModelPopulation)
  #print("hData:", hData)

  # MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElements)
  # blocksize = MaxOcup["BlockSize"]
  # gridsize = MaxOcup["GridSize"]  
  # gpCuda.compute_individuals[blocksize, gridsize](

  MaxOcup = gpCuda.gpuMaxUseProc(totalSemanticElements)
  blocks_per_grid = MaxOcup["GridSize"]
  threads_per_block = MaxOcup["BlockSize"]
  
  gpCuda.compute_individuals[blocks_per_grid, threads_per_block](
                      dModelPopulation,
                      dOutIndividuals,
                      dData,
                      numIndividuals,
                      GenesIndividuals,
                      nrowTrain,
                      nvar,
                      dStack,
                      dStackIdx,
                      1,
                      dStackModel,
                      dArrayTmp   
  )  

  #hOutIndividuals = dOutIndividuals.copy_to_host()
  #hStack = dStack.copy_to_host()
  #hStackIdx = dStackIdx.copy_to_host()
  hStackModel = dStackModel.copy_to_host()

  del hStack 
  del hStackIdx
  del hOutIndividuals 

  del dModelPopulation   
  del dData
  del dStack 
  del dStackIdx 
  del dStackModel 
  del dOutIndividuals
  gc.collect()

  return hStackModel
  
  