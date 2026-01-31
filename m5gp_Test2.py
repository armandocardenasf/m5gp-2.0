#Branch IF 3.8

from m5gp import m5gpRegressor as m5gp
import m5gpGlobals as gpG
#from   sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import sympy as sym
from sympy import symbols, Mul, simplify, count_ops

##import kagglehub
### Download latest version
##path = kagglehub.dataset_download("nikolasgegenava/sneakers-classification")
##print("Path to dataset files:", path)

#probando sympy
# Expresiones en forma de cadenas
expr1 = "X_0 + 2 * X_1"
expr2 = "X_2 - X_0"
expr3 = "X_1 * X_2 + 3"

# Multiplicar elemento a elemento
expr = expr1 +"," + expr2 + "," + expr3

expr1 = "((4848.926314903244)+((((((((((-16.081120787250963)*(978.0))+((26.799611267399413)*(188.0)))+((-10.334132305844975)*(-563.0)))+((-15.87872084319932)*(cos(X_6))))+((0.7041530790337818)*X_3))+((5.998255000797148e-15)*(exp((abs(X_1))))))+((0.048123253400739974)*X_0))+((-0.2417487831029137)*((3.0)/(abs(X_4)))))+(X_6+X_6)))"
print("\nExpresion 1:")
print(expr1)
print("\nComplejidad (número de operaciones):", count_ops(expr1))

print("\nExpresion 1 simplificada:")
D = simplify(expr1)
print(D)
print("\nComplejidad (número de operaciones):", count_ops(D))

mult_expr = "Mul(" + expr + ")"
mult = simplify(mult_expr)
print("\nProducto expandido:")
print(mult)

mult_expr = "Add(" + expr + ")"
mult = simplify(mult_expr)
print("\nSuma expandido:")
print(mult)

mult_expr = "Add(" + expr + ") / 3"  #+ str(count_ops(expr))
mult = simplify(mult_expr)
print("\nPromedio expandido:")
print(mult)

#arr1 = np.float32[:]
#print(type(arr1))

exit(0)

#load the data
#dataset = pd.DataFrame(pd.read_csv("/home/treelab/python-codes/data/Concrete/train_10107_1.txt" ,sep='\s+', header=None))
#dataset = pd.DataFrame(pd.read_csv("/home/acardenasf/pmlb/datasets5/589_fri_c2_1000_25/589_fri_c2_1000_25.tsv" ,sep='\s+', header=None))
dataset = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/test_10107_10.csv" ,sep=' ', header=None))

nrows = len(dataset.index)
nvar = dataset.shape[1] - 1
#print("Leyo X")
X = dataset.iloc[0:nrows, 0:nvar-1]
y = dataset.iloc[:nrows, nvar-1]

x_train = dataset.iloc[0:nrows, 0:nvar-1].to_numpy().astype(np.float32)
y_train = dataset.iloc[:nrows, nvar-1].to_numpy().astype(np.float32)
 
#X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.70,test_size=0.30,random_state=n)

print('Running m5gp ...')  
 
est = m5gp(
            generations=5, # number of generations (limited by default)
            Individuals=5, # number of individuals
            GenesIndividuals=20, # number of genes per individual
            mutationProb=0.1, # mutation rate probability
            mutationDeleteRateProb=0.01,  # mutation delete rate probality
            sizeTournament=0.15, # size of tournament
            evaluationMethod=2,  #error evaluation method 
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
            maxRandomConstant=999, #number of constants (-maxRandomConstant to maxRandomConstant)
            genOperatorProb=0.50, #probablity for generate Operators 
            genVariableProb=0.39, #probablity for generate variables 
            genConstantProb=0.1, #probablity for generate constants
            genNoopProb=0.01, #probablity for generate NOOP Operators 
			useOpIF=0, #Set if use IF operator
            log=1, #save log files
			verbose=1, #Show menssages on execution
            logPath='log/' #path for logs
 )

#Model = [-10099,  -1002,  -1005, -10005,    878,    647, -10007,  -1003,  -1001,   -1000]
#Model = [-10009, -10006, -10006,  -1000,  -1002,  -1005,   -737,    113,  -1005, -10007, -10002, -10004,  -1003, -10007,  -1005, -10009,  -1004, -10010, -1003, -549.]
Model = [-1002, -10009, -1000, -1002, -10001, -1003, -1001, -10002, -10012]

est.getStackExpr(Model)
#print("Complexity: ", est.complexity())
#print("Model: ",est.get_model())
exit(0)


