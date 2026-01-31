#Branch IF 3.8

from m5gp import m5gpRegressor as m5gp
import m5gpGlobals as gpG
#from   sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import sympy as sym
from sympy import symbols, Mul, simplify, count_ops
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

##import kagglehub
### Download latest version
##path = kagglehub.dataset_download("nikolasgegenava/sneakers-classification")
##print("Path to dataset files:", path)


#load the data
#dataset1 = pd.DataFrame(pd.read_csv("/home/treelab/python-codes/data/Concrete/train_10107_1.txt" ,sep='\s+', header=None))
#dataset1 = pd.DataFrame(pd.read_csv("/home/acardenasf/pmlb/datasets5/589_fri_c2_1000_25/589_fri_c2_1000_25.tsv" ,sep='/s+', header=None))
#dataset1 = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/test_10107_1.csv" ,sep=' ', header=None))
#dataset1 = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/207_autoPrice.tsv" ,sep='\t', header=None))
#dataset1 = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/344_mv.tsv" ,sep='\t', header=None))
#dataset1 = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/687_sleuth_ex1605.tsv" ,sep='\t', header=None))
#dataset1 = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/218_house_8L.tsv" ,sep='\t', header=None))

dsFile = "/home/acardenasf/datasets/1089_USCrime.tsv"

dataset1 = pd.DataFrame(pd.read_csv(dsFile ,sep='\t', header=None))

print("Leyo dataset:" , dsFile)
nrows = len(dataset1.index)
if (nrows > 10000):
    print("Hay mas de 10000")
    #dataset1 = dataset1.iloc[:10000]  #o df.head(10000)
    dataset1 = dataset1.sample(n=10000, random_state=42)

dataset = dataset1

nvar = dataset.shape[1] - 1
#print("Leyo X")
X = dataset.iloc[0:nrows, 0:nvar-1]
y = dataset.iloc[:nrows, nvar-1]

x_train = dataset.iloc[0:nrows, 0:nvar-1].to_numpy().astype(np.float32)
y_train = dataset.iloc[:nrows, nvar-1].to_numpy().astype(np.float32)

# # generate train/test split
# X_train, X_test, y_train, y_test = train_test_split(features, labels,
#                                                 train_size=0.75,
#                                                 test_size=0.25,
#                                                 random_state=random_state)

scaled = True
if (scaled): 
    print('scaling train X')
    sc_X = StandardScaler() 
    X_train_scaled = sc_X.fit_transform(x_train)

    print('scaling train y')
    sc_y = StandardScaler()
    y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()

    #Set train data (x, y)
    x_train = X_train_scaled
    y_train = y_train_scaled

#X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.70,test_size=0.30,random_state=n)

#functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs", "sum","prod", "avg", "std"]
#Operadores = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs"]
#functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "sqrt", "exp", "log", "abs"]
functions_set = ["+", "-", "*", "/", "sin", "cos", "sqrt", "exp", "log", "abs"]
#functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs"]

print('Running m5gp ...')  
 
est = m5gp( generations=30, # number of generations (limited by default) (40) (30)
            Individuals=640, # number of individuals (512) (256)
            GenesIndividuals=64, # number of genes per individual (64) (128)
            mutationProb=0.1, # mutation rate probability (0.1) (0.1)
            mutationDeleteRateProb=0.01,  # mutation delete rate probality (0.05) (0.01)
            sizeTournament=0.15, # size of tournament (0.15) (0.15)
            evaluationMethod=4,  #error evaluation method (2) (2)
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
            genOperatorProb=0.45, #probablity for generate Operators (0.45) (0.50)
            genVariableProb=0.40, #probablity for generate variables (0.40) (0.39)
            genConstantProb=0.05, #probablity for generate constants (0.05) (0.1)
            genNoopProb=0.1, #probablity for generate NOOP Operators (0.1) (0.01)
			useOpIF=0, #Set if use IF operator (0)
            functions_set = functions_set, # Set of operators for include into individuals 
            log=1, #save log files (1)
			verbose=1, #Show menssages on execution (1)
            logPath='log/' #path for logs
 )

# Model = [-10099,  -1002,  -1005, -10005,    878,    647, -10007,  -1003,  -1001,   -1000]
# Model =[-0.79660511, -10009, -10006, -10006,  -1000,  -1002,  -1005,   -737,    113,  -1005, -10007, -10002, -10004,  -1003, -10007,  -1005, -10009,  -1004, -10010, -1003, -549, -0.79660511]
# expr = est.getModelExpr(Model)
# print(Model)
# print(expr)
# exit(0)

#ea.cudacapabilities()

est.fit(x_train, y_train)

print("Complexity: ", est.complexity())
model = est.get_model()
print("Model: ",est.get_model())
#D = simplify(model)
#print(D)

yPredicted = est.predict(x_train)

# print('scaling test Y')
# sc_Y = StandardScaler() 
# X_train_scaled = sc_X.fit_transform(x_train)

# if (scaled): 
#     yPredicted = sc_y.inverse_transform(yPredicted)

#print("Y Data :\n", y_train)
#print("Y Predicted:\n", yPredicted)

mse = est.meanSquaredError(y_train, yPredicted)
print("mse: ", mse)
print("rmse:", est.rmse(y_train, yPredicted))
print ("R^2: ", est.R2(y_train, yPredicted))