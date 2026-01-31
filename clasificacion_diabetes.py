from m5gp import m5gpClassifier as m5gp  # usa Classifier para clasificación
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Cargar el dataset de Kaggle
#dataset_file = "/home/acardenasf/.cache/kagglehub/datasets/marshalpatel3558/diabetes-prediction-dataset/versions/1/diabetes_dataset.csv"
#dataset_file = "/home/luisarmando/.cache/kagglehub/datasets/marshalpatel3558/diabetes-prediction-dataset/versions/1/diabetes_dataset.csv"
dataset_file = "/home/acardenasf/.cache/kagglehub/datasets/saurabh00007/diabetescsv/versions/1/diabetes.csv"

dataset = pd.read_csv(dataset_file)
print("Columnas en el dataset:", dataset.columns)


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

print('Ejecutando M5GP en clasificación...')

# Instanciar el modelo de clasificacion
model = m5gp()

# Entrenar
model.fit(x_train, y_train)

# Evaluar en los datos de prueba
x_test = x_test.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)
predicciones = model.predict(x_test)

# Mostrar algunas predicciones
print("Predicciones:", predicciones[:10])
print("Reales:", y_test[:10])
