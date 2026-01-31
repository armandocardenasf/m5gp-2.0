# *********************************************************************
# Name: m5gpCumlMethods.py
# Description: Modulo que implementa las metodos para ejecutar multiples
# metodos de CuML
# Se utiliza la libreria cuml.
# *********************************************************************
 
import math
import time
import copy

try:
    import cupy as cp
    GPU_CUPY = True
except ImportError:
    GPU_CUPY = False

try:
    import cudf
    GPU_DF = True
except ImportError:
    GPU_DF = False

import gc
import numpy as np
import pandas as pd


#import rmm
#from rmm.allocators.cupy import rmm_cupy_allocator

try:
    import cuml as cu
    from cuml import LinearRegression
    from cuml.linear_model import LinearRegression
    from cuml import Ridge
    from cuml.linear_model import Ridge
    from cuml.linear_model import Lasso
    from cuml.kernel_ridge import KernelRidge
    from cuml.linear_model import ElasticNet

    from cuml import LogisticRegression
    from cuml import SVC
    from cuml import RandomForestClassifier
    from cuml import KNeighborsClassifier
    from cuml import MBSGDClassifier

    from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor

    # import metrics
    from cuml.metrics.regression import mean_squared_error as cuMSE
    from cuml.metrics.regression import r2_score as cuR2

    from cuml.metrics import accuracy_score
    #from cuml.metrics.accuracy import accuracy_score 
    from cuml.metrics import roc_auc_score 
    from sklearn.metrics import f1_score
    #from cuml.metrics import f1_score
    from sklearn.metrics import average_precision_score
    #from cuml.metrics import average_precision_score
    #from sklearn.preprocessing import StandardScaler
    from cuml.model_selection import train_test_split
    from cuml.preprocessing import StandardScaler
    GPU_CUML = True
except ImportError:
    GPU_CUML = False



from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import cpu_count
from multiprocessing import Manager

#import skcuda.cublas as cublas
#import pycuda.gpuarray as gpuarray

import m5gpGlobals as gpG


def cuGetMethodNameClassification(self):
        global cuMethod

        if self.evaluationMethod == 0:
            cuMethod = "Logistic Regression"
        elif self.evaluationMethod == 1:
            cuMethod = "Support Vector Classifier"
        elif self.evaluationMethod == 2:
            cuMethod = "Random Forest Classifier"
        elif self.evaluationMethod == 3:
            cuMethod = "K Neighbors Classifier"
        elif self.evaluationMethod == 4:
            cuMethod = "Mini Batch Classifier"

        return cuMethod

def getDefaultParams(evaluationMethod):

    if evaluationMethod == 0: # Logistic Regression 
            defaultParams = {
                "penalty": "l2",  # Default value
                "tol": 1e-4,  # Default value
                "C": 0.5,  # Default value
                "fit_intercept": True,  # Default value
                "class_weight": None,  # Default value
                "max_iter": 5000,  # Default value
                "linesearch_max_iter": 50,  # Default value
                "verbose": False,  # Default value
                "l1_ratio": None,  # Default value
                "solver": "qn",  # Default value
                "output_type": None
            }

    elif evaluationMethod == 1: # Support Vector Classifier
        defaultParams = {
                "C": 92.5,  # Default value
                "kernel": "rbf",  # Default value
                "degree": 2,  # Default value
                "gamma": "scale",  # Default value (auto)
                "coef0": 5.9,  # Default value
                "tol": 1e-3,  # Default value
                "cache_size": 1024.0,  # Default value
                "max_iter": -1,  # Default value
                "nochange_steps": 1000,  # Default value
                "verbose": False,  # Default value
                "output_type": None,  # Default value
                "class_weight" : 'balanced'
            }

    elif evaluationMethod == 2: # Random Forest Classifier
        defaultParams = {
                "n_estimators": 200,  # Default value
                "split_criterion": 0,  # Default value
                "bootstrap": True,  # Default value
                "max_samples": 1.0,  # Default value
                "max_depth": 16,  # Default value
                "max_leaves": -1,  # Default value
                #"max_features": "auto",  # Default value
                "max_features": "sqrt",
                "n_bins": 128,  # Default value
                "n_streams": 4,  # Default value
                "min_samples_leaf": 1,  # Default value
                "min_samples_split": 2,  # Default value
                "min_impurity_decrease": 0.0,  # Default value
                "max_batch_size": 4096,  # Default value
                "random_state": None,  # Default value
                "verbose": False,  # Default value
                "output_type": None  # Default value
            }

    elif evaluationMethod == 3: # K Neighbors Classifier
        defaultParams = {
                "n_neighbors": 5,  # Default value
                "algorithm": "auto",  # Default value
                "metric": "euclidean",  # Default value
                "weights": "uniform",  # Default value
                "verbose": False,  # Default value
                "output_type": None  # Default value
            }
    elif evaluationMethod == 4: # Mini Batch Classifier
        defaultParams = {
                "loss": 'hinge',
                "penalty": "l2",
                "alpha": 0.0001,
                "l1_ratio": 0.15,
                "batch_size": 32,
                "fit_intercept": True,
                "epochs": 1000,
                "tol": 1e-3,
                "shuffle": True,
                "eta0": 0.001,
                "power_t": 0.5,
                "learning_rate": 'constant',
                "n_iter_no_change": 5,
                "verbose": False,
                "output_type": None
            }
    
    return defaultParams


def validateParameters(params,evaluationMethod):
    defaultParams = getDefaultParams(evaluationMethod)

    keys_dict1 = set(params.keys())
    keys_dict2 = set(defaultParams.keys())

    if (keys_dict1.issubset(keys_dict2)):

        for key, value in params.items():
            defaultParams[key] = value
    else: 
        print("Check your params, One or more are incorrect")
        exit()
    
    return defaultParams


def createCumlMethodClassification(evaluationMethod, params=None):
    if params== None:
        params = getDefaultParams(evaluationMethod)
    if evaluationMethod == 0 :
        slr = LogisticRegression(**params)
    if evaluationMethod == 1:
        slr = SVC(**params, probability= True)
    if evaluationMethod == 2:
        slr = RandomForestClassifier(**params)
    if evaluationMethod == 3:
        slr = KNeighborsClassifier(**params)
    if evaluationMethod == 4: # es muy lento
        slr = MBSGDClassifier()
    return slr

def ExecCumlClassification(nProc, hFit,  st, evaluationMethod, indiv, genes, nrows, hStackIdx, y_train, scorer, 
                            params , CrossVal, k, averageMode, 
                            CrossAverage, slr):

    fit = 0
    if (nProc > indiv):       
        return
    
    # Desactivar los mensajes de registro
    #logging.getLogger('cuml').setLevel(logging.ERROR)
    
    #slr = createCumlMethodClassification(evaluationMethod, params)
    #slr = copy.deepcopy(slr1)

    ind = st[nProc]

    #como los datos vienen como vector, lo convertimos como matriz de cupy
    ind2 = ind.reshape(nrows, genes)

    #Obtenemos el numero de columnas (elementos del stack)
    tt = int(hStackIdx[nProc*nrows])

    # Transformamos como matriz el vector del individuo obtenido del stack 
    sX_train = ind2[:, :tt]      

    sCols = sX_train.shape[1]
    cX = cudf.DataFrame()
    cY = cudf.DataFrame()

    if (sCols >= 1) :
        cX = cp.asarray(sX_train, dtype=cp.float32)
        cY = cp.asarray(y_train, dtype=cp.float32)

        # print("Checa NAN/INF")
        # # Validar NaN
        # if cp.isnan(cX).any():
        #     print("cX contiene valores NaN")

        # if cp.isnan(cY).any():
        #     print("cY contiene valores NaN")

        # # Validar Inf o -Inf
        # if cp.isinf(cX).any():
        #     print("cX contiene valores Inf")

        # if cp.isinf(cY).any():
        #     print("cY contiene valores Inf")

        cX = cp.nan_to_num(cX, nan=0.0, posinf=0.0, neginf=0.0)
        cY = cp.nan_to_num(cY, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) escalar cX
        # scaler = StandardScaler(with_mean=True, with_std=True)
        # cX = scaler.fit_transform(cX)

        if CrossVal:
            #print("CrossValidation")
            fit, slr =  CrossValidation(slr, cX, cY, scorer, k, averageMode, CrossAverage)
        else:
            #print("Fit")
            # Procesamos el Fit con el arreglo transformado
            reg = slr.fit(cX, cY)

            yPred = make_predictions(slr,scorer,cp.asnumpy(cX))

            cuModel= copy.deepcopy(slr)

            fit = evaluationMetrics(scorer,cp.asnumpy(cY),yPred,averageMode)

        cuModel = copy.deepcopy(slr)
    else :      
        fit = 0
        cuModel = copy.deepcopy(slr)
    #endif

    if math.isnan(fit) or math.isinf(fit):
        fit = 0
     
    
    hFit[nProc] = fit
    
    cX = []
    cY = []
    sX_train = []
    return cuModel

def EvaluateCuml2Classification(self, hStack, hStackIdx, hFit, y_train) :
    global cuMethod
    global cuModel
    global slr

    cuModel = []
                    
    #Obtenemos todos los stack con todos los resultados de la matriz semantica
    st = hStack.reshape(self.Individuals, self.nrowTrain * self.GenesIndividuals)

    print("EvaluateCuml2Classification:", len(st))
    slr = createCumlMethodClassification(self.evaluationMethod, self.params)
    #Ejecuta la evaluacion de CUML de manera secuencial
    for i in range(self.Individuals):
        #print(f"Individual: {i}")
        hRes = ExecCumlClassification(i, hFit, st, self.evaluationMethod, self.Individuals, self.GenesIndividuals, self.nrowTrain, hStackIdx, y_train, self.scorer, self.params, self.crossVal, self.k, self.averageMode, self.CrossAverage, slr)
        
        # Regresa el modelo de CUML del individuo generado (hRes)
        slr2 = copy.deepcopy(hRes)

        # Agregamos el modelo CUML del individuo en un arreglo
        cuModel.insert(i,slr2)

    return hFit, cuModel


def evaluationMetrics(scorer ,y_true, y_pred, averageMode):
    classes = pd.DataFrame(y_true)[0].unique().size
    average = 'binary'
    
    if(classes > 2 ):
        if (averageMode == ["micro", "macro", "weighted", "samples"]):
            average = averageMode
        else:
            average = "macro"

    if (scorer == 0): #0: Accuracy Score (accuracy de cuML)
            fit = accuracy_score(y_true, y_pred)      
    elif (scorer == 1): #1: ROC AUC Score (cuROCAUC de cuML)
        fit = roc_auc_score(y_true, y_pred)
    elif (scorer == 2): # 2: F1 Score (f1_score de scikit-learn)
        fit = f1_score(y_true, y_pred, average = average)
        #fit = f1_multiclass_numba_cuda(y_true, y_pred, average=average)
    elif (scorer == 3): #  3: Average Precision Score (average_precision_score de scikit-learn)
        fit = average_precision_score(y_true, y_pred, average = averageMode)
    return fit

def make_predictions(slr,scorer,X):
    if (scorer != 3):
        y_pred = slr.predict(X)
    else:
        y_pred = slr.predict_proba(X)
    return y_pred

def CrossValidation(slr, cX , cY, scorer, k, averageMode, CrossAverage):
    # Load your data into a cuDF dataframe
    X = cudf.DataFrame(cX)  # Your features
    y = cudf.Series(cY)     # Your target variable

    scores = []    

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0 )

    # Shuffle the data
    shuffle_indices = np.random.permutation(len(X_train))
    #print(" shuffle_indices:",  shuffle_indices)
    X_train_shuffled = X_train.iloc[shuffle_indices]
    y_train_shuffled = y_train.iloc[shuffle_indices]


    # Determine the size of each fold
    fold_size = len(X_train) // k

    bestCrossScore = 0
    bestModel = 0
    
    # Perform k-fold cross-validation
    for i in range(k):
        # Determine the start and end indices for the current fold
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else len(X_train)

        # Split the data into training and validation sets for this fold
        X_val_fold = X_train_shuffled[start:end]
        y_val_fold = y_train_shuffled[start:end]
        X_train_fold = cudf.concat([X_train_shuffled[:start], X_train_shuffled[end:]])
        y_train_fold = cudf.concat([y_train_shuffled[:start], y_train_shuffled[end:]])
        

        # 2) Limpiar Inf/-Inf en cuDF (GPU) → NaN
        X_train_fold = X_train_fold.replace([cp.inf, -cp.inf], cp.nan)
        y_train_fold = y_train_fold.replace([cp.inf, -cp.inf], cp.nan)

        # 3) Llenar NaN con 0 (o algún valor neutro que tú decidas)
        X_train_fold = X_train_fold.fillna(0)
        y_train_fold = y_train_fold.fillna(0)

        # 4) Convertir a NumPy en float64
        X_train_np = X_train_fold.to_numpy(dtype=np.float64)
        y_train_np = y_train_fold.to_numpy().astype(np.int64)  # asumiendo etiquetas enteras

        # 5) Clip para evitar valores extremos que provocan overflow en el kernel
        X_train_np = np.clip(X_train_np, -1e3, 1e3)

        # 6) Escalar (muy importante para SVM)
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train_np)

        if np.unique(y_train_fold.to_numpy()).size < 2:
            print("Fold con una sola clase, se omite este fold")
            continue

        # Train the model on the training fold
        slr.fit(X_train_np, y_train_np)
        #slr.fit(X_train_fold.to_numpy(), y_train_fold.to_numpy())
        
        # Evaluate the model on the validation fold
        y_pred = make_predictions(slr,scorer,X_val_fold.to_numpy())
        y_pred_training = make_predictions(slr,scorer,X_train_np)        
        # y_pred = make_predictions(slr,scorer,X_val_fold.to_numpy())
        # y_pred_training = make_predictions(slr,scorer,X_train_fold.to_numpy())
    

        if np.isnan(y_pred).any():
            nan_indices = np.isnan(y_pred)
            y_pred[nan_indices] = 0
        if np.isnan(y_pred_training).any():
            nan_indices = np.isnan(y_pred_training)
            y_pred_training[nan_indices] = 0
        

        score = evaluationMetrics(scorer, y_val_fold.to_numpy(), y_pred, averageMode)
        scoreTrain = evaluationMetrics(scorer, y_train_fold.to_numpy(), y_pred_training, averageMode)

        if math.isnan(score) or math.isinf(score):
            score = 0.01
        #print(f"Training fit ({i}): {scoreTrain} Validation score: {score}")
        if score > bestCrossScore:
            bestCrossScore = score
            bestModel = copy.deepcopy(slr)
        scores.append(score)
    #end for
    
    y_test_pred = make_predictions(bestModel, scorer, X_test.to_numpy())
    if np.isnan(y_test_pred).any():
        nan_indices = np.isnan(y_test_pred)
        y_test_pred[nan_indices] = 0
    
    if CrossAverage == False:
        test_score = evaluationMetrics(scorer, y_test.to_numpy(), y_test_pred, averageMode)
    else:
        test_score = np.mean(scores)
        print(f"FinaL score cross validation: {test_score}")
    return test_score, bestModel





