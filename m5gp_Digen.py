
#Import a package and initialize the benchmark
from digen import Benchmark
import pandas as pd
import logging
import optuna
import pickle
import matplotlib.pyplot as plt
import argparse
import os
import sys, getopt
from pathlib import Path

  
RDir = ''
onlyGraph = False
n = len(sys.argv)

######

# Desactivar los mensajes de registro
logging.getLogger('cuml').setLevel(logging.ERROR)
benchmark = Benchmark()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default=None, help="Specify a dataset (otherwise all datasets are used)",
                required=False, nargs='?')
parser.add_argument("-r", "--rPath", default=None, help="Specify a results path (otherwise Results are used)",
                required=False, nargs='?')
parser.add_argument("-g", "--graphs", default=None, help="Specify if only generate graphs",
                required=False, nargs='?')
args = parser.parse_args()


datasets = args.dataset
if args.dataset is None:
        datasets = benchmark.list_datasets()
        print("All databases")
else :
        print(f"Database: {datasets}")
RDir = args.rPath
if args.rPath is None:
        RDir = 'Results'
print(f"Path results: {RDir}")

pGraph = args.graphs
print(pGraph)
if args.graphs is None:
        onlyGraph = False        
        print("Executing Benchmark")
else :
        onlyGraph = True
        print("Only generating graphs")
#######


# Usando pathlib (más moderno y recomendado) ---
path_results = Path(RDir)
path_str = str(path_results) 

# crea el directorio si no existe. exist_ok=True evita errores si ya existe.
path_results.mkdir(parents=True, exist_ok=True)
print(f"Directorio '{path_results}' asegurado (creado si no existía).")


if (onlyGraph == False) :
        from m5gp import m5gpClassifier

        print('Ejecutando M5GP para clasificación...')
        # Instanciar el modelo de clasificacion
        est = m5gpClassifier()

        def parameters(trial):
                functions_set = ["+", "-", "*", "/", "sin", "cos", "sqrt", "exp", "log", "abs"]

                params = {
                        'generations':1, 
                        'Individuals':16, # (32)
                        'GenesIndividuals':1024, #(1024) 
                        'mutationProb':0.1, 
                        'mutationDeleteRateProb':0.01,  
                        'evaluationMethod':0,  # error evaluation method (2) (2)
                                # - 0: Logistic Regression
                                # - 1: Support Vector Classifier
                                # - 2: Random Forest Classifier
                                # - 3: K Neighbors Classifier
                        'scorer':0,  
                        'sizeTournament':0.15, 
                        'maxRandomConstant':1, 
                        'genOperatorProb':0.50, 
                        'genVariableProb':0.39, 
                        'genConstantProb':0.1, 
                        'genNoopProb':0.01,  
                        'useOpIF':0,   
                        'log':1, 
                        'functions_set' : functions_set, 
                        'verbose':1, 
                        'logPath':'log/',
                        'crossVal' : True,
                        'k' : 3 ,
                        'averageMode' : "macro",
                        'CrossAverage' : False,
                        'params':None
                }
                return params

        # Perform optimizations on each of the DIGEN datasets in order to 
        # fairly compare its performance against predefined methods
        results = benchmark.optimize(est = est, 
                                datasets='digen30',
                                parameter_scopes = parameters, 
                                storage='sqlite:///' + path_str + '/DigenDatasets.db')
        pickle.dump( results, open( path_str + '/DigenDatasets'+".pkl", "wb" ) )

#Print
print('Generating results graphs')

#Let's skipp this, and load the previous results
results=pickle.load(open(path_str + '/DigenDatasets'+".pkl","rb"))

print(results)
benchmark.plot_heatmap(new_results=results)
plt.savefig('GeneralHeatMap.png', dpi=300, bbox_inches='tight')  # dpi for resolution, bbox_inches for removing extra whitespace
        



