import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 500)
y = x**2

# se pasa una función matemática
# se puede customizar añadiendo título, ejes x e y, rejilla..
plt.plot(x, y, '-.k')
plt.title("Mi gráfico")
plt.xlabel("x")
plt.ylabel("y = x**2")
plt.grid(True)
plt.show()


# diamantes rojos
x2 = np.linspace(-2, 2, 20)
y2 = x2**3
# se añade el primer gráfico para mostrar ambos en una misma gráfica
# se añade una etiqueta para mostrar una leyenda
plt.plot(x, y, '-.k', label="y = x**2")
plt.plot(x2, y2, 'dr', label="y = x**3")
plt.title("Mi gráfico")
# se puede cambiar el tamaño de la fuente con fontsize
plt.xlabel("x", fontsize=18)
plt.ylabel("y = x**3", fontsize=18)
plt.grid(True)
plt.legend(loc='lower right', fontsize=14)
# delimita la extensión de los ejes
plt.axis([-2,2,-4,4])
# muestra o guarda el gráfico.
plt.show()
#plt.savefig("mi_grafico.png", transparent=True)


# Múltiples gráficos


from sklearn.datasets import fetch_openml

# descargamos el dataset MNIST
mnist = fetch_openml('mnist_784', version=1)
# X = datos de las imágenes
# Y = contiene las etiquetas (0,9)
X, y = mnist["data"], mnist["target"]

print(X.shape, y.shape)

# convertir X a array antes del reshape
X = np.array(X)
# convertimos la imagen de 784p al tamaño original de 28p
X = X.reshape(-1, 28,28)

print(X.shape)


# FIGURAS

# se define una figura y añadimos dos más por columna
fig = plt.figure(figsize=(3,3))
ax = plt.subplot(1,3,1)
ax.imshow(X[0], cmap="gray")
plt.axis('off')
ax = plt.subplot(1,3,2)
ax.imshow(X[1], cmap="gray")
ax.axis('off')
ax = plt.subplot(1,3,3)
ax.imshow(X[2], cmap="gray")
ax.axis('off')
plt.show()


# SCATTER

from numpy.random import rand

# muestra 100 puntos aleatorios entre 0 y 1
# dando estos tres colores aleatorios con diferentes tamaños
for color in ['red', 'green', 'blue']:
    n = 100
    x, y = rand(2, n)
    scale = 500.0 * rand(n) ** 5
    plt.scatter(x, y, s=scale, c=color, alpha=0.3, edgecolors='blue')
plt.grid(True)
plt.show()


# HISTOGRAMAS

# se muestra los datos de data, número de barras con bins y ancho de esta con rwidth
data = [1, 1.1, 1.8, 2, 2.1, 3.2, 3, 3, 3, 3]
plt.hist(data, bins = 10, rwidth=0.8)
plt.show()