
from .src.m5gp import m5gp
import pandas as pd
 
hyper_params = [
        {
            'generations' : (30,),
            'Individuals' : (128,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.25,),
        }, 
        {
            'generations' : (30,),
            'Individuals' : (128,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.15,),
        },
        {
            'generations' : (30,),
            'Individuals' : (128,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.1,),
        },
        {
            'generations' : (30,),
            'Individuals' : (128,),
            'GenesIndividuals' : (256,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.1,),
        },
        {
            'generations' : (30,),
            'Individuals' : (256,),
            'GenesIndividuals' : (256,),
            'mutationProb' : (0.15,),
            'sizeTournament' : (0.15,),
        },
        {
            'generations' : (50,),
            'Individuals' : (128,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.25,),
        },    
        {
            'generations' : (50,),
            'Individuals' : (256,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.15,),
        },                                  
    ]

#functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs", "sum","prod", "avg", "std"]
#functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs"]
functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "sqrt", "exp", "log", "abs"]

# from .src.m5gp import m5gp
# from m5gp import m5gpRegressor as m5gp

# Create the pipeline for the model
print('Running m5gp ...')
est = m5gp.m5gpRegressor( 
            generations=30, # number of generations (limited by default) (40) (30)
            Individuals=256, # number of individuals (512) (256)
            GenesIndividuals=128, # number of genes per individual (64) (128)
            mutationProb=0.1, # mutation rate probability (0.1) (0.1)
            mutationDeleteRateProb=0.01,  # mutation delete rate probality (0.05) (0.01)
            sizeTournament=0.15, # size of tournament (0.15) (0.15)
            evaluationMethod=2,  #error evaluation method (2) (2)
                        # 0=RMSE, 
                        # 1=R2, 
                        #cuML Methods
                        # 2=LinearRegression, 3=Lasso Regression, 
                        # 4=Ridge regression, 5=kernel Ridge Regression,
                        # 6=ElasticNet Regression
                        #cuML MiniBatch options
                        # 7=MiniBatch none regularization (linear regression)
                        # 8=MiniBatch lasso regularization 
                        # 9=MiniBatch ridge regularization 
                        #10=MiniBatch elasticnet regularization 
            scorer=0, #Compute Error using: 0/1 => RMSE, 2 => R2 (0)
            maxRandomConstant=1, #number of constants (-maxRandomConstant to maxRandomConstant) (1)
            genOperatorProb=0.50, #probablity for generate Operators (0.45) (0.50)
            genVariableProb=0.39, #probablity for generate variables (0.40) (0.39)
            genConstantProb=0.1, #probablity for generate constants (0.05) (0.1)
            genNoopProb=0.01, #probablity for generate NOOP Operators (0.1) (0.01)
			useOpIF=0, #Set if use IF operator (0)
            functions_set = functions_set, # Set of operators for include into individuals 
            log=1, #save log files (1)
			verbose=1, #Show menssages on execution (1)
            logPath='log/' #path for logs
 )

def complexity(est):
    print("Complexity:", est.get_n_nodes())
    nodes = est.get_n_nodes()
    return nodes

def model(est):
    indiv = est.best_individual()
    return str(indiv)
