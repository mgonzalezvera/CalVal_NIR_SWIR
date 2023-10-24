import numpy as np
import matplotlib.pyplot as plt

# Cargar la máscara desde un archivo CSV
loaded_matrix = np.load('/home/usuario/Documentos/MISION/CalVal/arrays/dead_pixels_mask.npy')

# Crear una figura y un eje
fig, ax = plt.subplots()

# Configurar colores (True: blanco, False: negro)
cmap = plt.cm.gray  # Colores en escala de grises

# Mostrar la máscara en el gráfico
ax.imshow(loaded_matrix, cmap=cmap)

# Configurar el aspecto de la gráfica
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])

# Mostrar el gráfico
plt.show()