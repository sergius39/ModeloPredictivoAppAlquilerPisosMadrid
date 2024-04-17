import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Dataset
'''
Predecir el precio de una vivienda en la comunidad de Madrid en funcion de una serie de caracteristicas:
1.Zona 0 (Carabanchel), 1 (Villa Vallecas), 2(Usera), 3 (Arganzuela), 4 (Latina), 5 (Ciudad lineal), 6 (Castellana), 7 (Retiro), 8 (Goya),
2. m^2
3. nº habitaciones
4. nº planta
5. Ascensor 0 (no), 1 (si)
6. Vivienda exterior 0 (no), 1 (si)
7. Estado 0 (no rehabilitado), 1(rehabilitado), 2(Nuevo)
8. Vivienda centrica 0 (no), 1 (si)

Salida: Precio de la vivienda

'''

datos = [
    [0, 50, 1, 2, 1, 1, 0, 0],
    [0, 70, 2, 4, 1, 1, 1, 0],
    [0, 80, 3, 4, 0, 1, 1, 0],
    [0, 75, 4, 3, 1, 0, 1, 0],
    [1, 45, 1, 4, 1, 1, 0, 0],
    [1, 81, 2, 6, 1, 1, 2, 0],
    [1, 78, 2, 2, 1 ,1, 2, 0],
    [1, 62, 2, 2, 1, 1, 0, 0],
    [1, 105, 3, 5, 1, 1, 0, 0],
    [2, 25, 0, 1, 0, 1, 1, 0],
    [2, 65, 2, 1, 0, 1, 0, 0],
    [2, 105, 3, 1, 0, 1, 0, 0],
    [3, 40, 1, 0, 0, 1, 0, 1],
    [3, 44, 1, 5, 1, 1, 0, 1],
    [3, 50, 1, 2, 1, 1, 0, 1],
    [3, 93, 3, 5, 1, 1, 0, 1],
    [4, 75, 3, 3, 1, 1, 0, 1],
    [4, 53, 2, 5, 0, 1, 1, 1],
    [4, 90, 3, 2, 1, 1, 1, 1],
    [5, 49, 2, 1, 0, 1, 0, 1],
    [5, 29, 0, 0, 1, 1, 1, 1],
    [5, 125, 3, 1, 1, 1, 0, 1],
    [6, 110, 2, 2, 1, 1, 1, 1],
    [6, 267, 4, 4, 1, 1, 1, 1],
    [6, 203, 5, 5, 1, 1, 0, 1],
    [6, 68, 3, 5, 1, 0, 1, 1],
    [6, 46, 1, 0, 1, 0, 0, 1],
    [7, 40, 1, 3, 0, 0, 0, 1],
    [7, 92, 2, 0, 1, 0, 1, 1],
    [7, 104, 4, 4, 1, 1, 1, 1],
    [7, 150, 5, 7, 1, 1, 1, 1],
    [8, 57, 2, 3, 1, 0, 0, 1],
    [8, 184, 3, 3, 1, 1, 1, 1],
    [8, 150, 5, 1, 1, 1, 0, 1]

]

precios = [830, 1100, 1150, 1475, 820, 1164, 1329, 955, 1220, 600, 850, 1200, 750, 850, 1100, 1350, 1200, 890, 1100, 930, 650, 1600, 3700, 10000, 4600, 2450, 1100, 825, 2850, 1800, 3950, 1600, 4495, 3500]

datos = np.array(datos)
precios = np.array(precios)

entrada = tf.keras.layers.Dense(units=8, input_shape=[8])
oculta1 = tf.keras.layers.Dense(units=8)
oculta2 = tf.keras.layers.Dense(units=4)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([entrada, oculta1, oculta2, salida])

modelo.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss = "mean_squared_error"
)

print("Entrenando el modelo...")
historial = modelo.fit(datos, precios, epochs=1000, verbose=False)
print("Modelo entrenado!!!")

plt.xlabel('numIntentos')
plt.ylabel('Perdida')
plt.plot(historial.history['loss'])
plt.show()

modelo.save('appAlquilerCasas.h5')
