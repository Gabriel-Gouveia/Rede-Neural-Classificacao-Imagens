from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('personagens.csv')

X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values
y = (y == 'Bart')

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    X, y, test_size=0.2)

rede_neural = tf.keras.models.Sequential()

rede_neural.add(tf.keras.layers.Dense(
    units=4, activation="relu", input_shape=(6,)))

rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu'))
rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu'))
rede_neural.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

rede_neural.compile(
    optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

historico = rede_neural.fit(
    X_treinamento, y_treinamento, epochs=50, validation_split=0.1)

plt.plot(historico.history['val_loss'])

plt.plot(historico.history['val_accuracy'])

previsoes = rede_neural.predict(X_teste)
print(previsoes)

previsoes = (previsoes > 0.5)
print(previsoes)


print(y_teste)

from sklearn.metrics import accuracy_score

print(accuracy_score(previsoes, y_teste))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_teste, previsoes)

print(cm)

sns.heatmap(cm, annot=True)