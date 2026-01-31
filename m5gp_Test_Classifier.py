#Branch IF 3.8

from m5gp import m5gpClassifier as m5gp  # usa Classifier para clasificación
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Cargar el dataset de Kaggle
#dataset_file = "/home/acardenasf/.cache/kagglehub/datasets/marshalpatel3558/diabetes-prediction-dataset/versions/1/diabetes_dataset.csv"
#dataset_file = "/home/luisarmando/.cache/kagglehub/datasets/marshalpatel3558/diabetes-prediction-dataset/versions/1/diabetes_dataset.csv"
dataset_file = "/home/acardenasf/.cache/kagglehub/datasets/saurabh00007/diabetescsv/versions/1/diabetes.csv"

dataset = pd.read_csv(dataset_file)
print("Columnas en el dataset:", dataset.columns)

print("Leyo dataset:" , dataset_file)
nrows = len(dataset.index)
if (nrows > 10000):
    print("Hay mas de 10000")
    #dataset1 = dataset1.iloc[:10000]  #o df.head(10000)
    dataset = dataset.sample(n=10000, random_state=42)

# Verifica los nombres de columnas si es necesario
# print(dataset.columns)

# Suponiendo que la columna objetivo se llama 'class' o 'Outcome':
X = dataset.drop(columns=["Outcome"])  # Todas menos la etiqueta
y = dataset["Outcome"]                # Etiqueta/clase

# Separar en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Convertir a float32 para M5GP
x_train = x_train.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)

# scaled = True
# if (scaled): 
#     print('scaling train X')
#     sc_X = StandardScaler() 
#     X_train_scaled = sc_X.fit_transform(x_train)

#     # print('scaling train y')
#     # sc_y = StandardScaler()
#     # y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()

#     #Set train data (x, y)
#     x_train = X_train_scaled
#     # y_train = y_train_scaled

functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "sqrt", "exp", "log", "abs"]
functions_set = ["+", "-", "*", "/", "sin", "cos", "sqrt", "exp", "log", "abs"]

print('Ejecutando M5GP en clasificación...')

# Instanciar el modelo de clasificacion
model = m5gp(
        generations=1, 
        Individuals=16, # (32)
        GenesIndividuals=1024, 
        mutationProb=0.1, 
        mutationDeleteRateProb=0.01,  
        evaluationMethod=0,  # error evaluation method (2) (2)
                # - 0: Logistic Regression
                # - 1: Support Vector Classifier
                # - 2: Random Forest Classifier
                # - 3: K Neighbors Classifier
        scorer=0,  
        sizeTournament=0.15, 
        maxRandomConstant=1, 
        genOperatorProb=0.50, 
        genVariableProb=0.39, 
        genConstantProb=0.1, 
        genNoopProb=0.01,  
        useOpIF=0,   
        log=1, 
        functions_set = functions_set, 
        verbose=1, 
        logPath='log/',
        function_set = '',
        crossVal = True,
        k = 3 ,
        averageMode = "macro",
        CrossAverage = False,
        params=None
)

# Entrenar
model.fit(x_train, y_train)

# Evaluar en los datos de prueba
x_test = x_test.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)
predicciones = model.predict(x_test)

# Mostrar algunas predicciones
print("Predicciones:", predicciones[:10])
print("Reales:", y_test[:10])
