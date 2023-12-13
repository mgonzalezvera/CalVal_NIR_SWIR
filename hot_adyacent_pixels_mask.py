# import matplotlib.pyplot as plt
# import numpy as np
# import mat73
# from scipy.ndimage import label
#
# # Paso 1: Calcular la media de toda la imagen
# # path = '/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/' #path CONAE
# path = '/media/maxpower/Mauro/SABIA-mar/20230918_NirSwir_INVAP/' #path CASA
#
# file =path+'Darks/Temp2/dark_temp2_20230920105004.352.mat'
#
# df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
# media_imagen = np.mean(df)
# print(media_imagen)
# # Paso 2: Comparar valores con la media y guardar puntos que superen el THRESHOLD
# THRESHOLD_PERCENT = 200  # Porcentaje del valor de la media como threshold
# threshold_value = media_imagen * (THRESHOLD_PERCENT / 100)
# coordenadas = np.loadtxt('./arrays/coordenadas_dead_pixels.csv', delimiter=',', dtype=int)
#
# hot_pixels_ad = []
# for y, x in coordenadas:
#     # Verificar puntos adyacentes
#     for i in range(-1, 2):
#         for j in range(-1, 2):
#             if 0 <= y + i < df.shape[0] and 0 <= x + j < df.shape[1]:
#                 if df[y + i, x + j] > threshold_value:
#                     hot_pixels_ad.append((y + i, x + j))
#
# # Paso 3: Eliminar puntos repetidos en hot_pixels_ad
# # hot_pixels_ad = list(set(hot_pixels_ad))
# hot_pixels_ad = sorted(list(set(hot_pixels_ad)))
#
# # Paso 4: Plotear los puntos en azul con un círculo y etiquetas
# fig, axs = plt.subplots()
# vmin= 60
# vmax=1000
# im = axs.imshow(df, cmap='viridis', vmin=vmin, vmax=vmax)
# axs.set_title('T_int = {} ms'.format(mat73.loadmat(file)['salida']['tiemposInt']))
#
# # Marcar píxeles adyacentes a coordenadas
# for i, (y, x) in enumerate(coordenadas):
#     axs.plot(x, y, 'ro', markersize=2)
#     axs.annotate('{}'.format(i + 1), (x, y), color='r', fontsize=12)
#
# # Marcar píxeles adyacentes que superan el threshold
# for i, (y, x) in enumerate(hot_pixels_ad):
#     axs.plot(x, y, 'bo', markersize=2)
#     axs.annotate('{}'.format(i + 1), (x, y), color='b', fontsize=12)
#
# # Dibujar círculos alrededor de los puntos
# # for (y, x) in hot_pixels_ad:
# #     circle = plt.Circle((x, y), 5, color='b', fill=False)
# #     axs.add_patch(circle)
#
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import mat73

def plot_hot_pixels(file, threshold_percent, axs, color,off):
    # Calcular la media de toda la imagen
    df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
    media_imagen = np.mean(df)

    # Comparar valores con la media y guardar puntos que superen el THRESHOLD
    threshold_value = media_imagen * (threshold_percent / 100)
    coordenadas = np.loadtxt('./arrays/coordenadas_dead_pixels.csv', delimiter=',', dtype=int)

    hot_pixels_ad = []
    for y, x in coordenadas:
        # Verificar puntos adyacentes
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= y + i < df.shape[0] and 0 <= x + j < df.shape[1]:
                    if df[y + i, x + j] > threshold_value:
                        hot_pixels_ad.append((y + i, x + j))

    # Eliminar puntos repetidos en hot_pixels_ad
    hot_pixels_ad = sorted(list(set(hot_pixels_ad)))

    # Marcar píxeles adyacentes a coordenadas
    for i, (y, x) in enumerate(coordenadas):
        axs.plot(x, y, 'ro', markersize=2)
        axs.annotate('{}'.format(i + 1), (x, y), color='r', fontsize=16)

    # Marcar píxeles adyacentes que superan el threshold
    offset_x = 0.3 * (off % 3)  # Ajustar según tus necesidades
    offset_y = 0.3 * (off % 3)  # Ajustar según tus necesidades
    for i, (y, x) in enumerate(hot_pixels_ad):
        axs.plot(x, y, 'o', markersize=2, label=f'Threshold {threshold_percent}%', color=color)
        axs.annotate('{}'.format(i + 1), (x + offset_x, y + offset_y), fontsize=16, color=color)


# Crear un gráfico para cada valor de threshold en el mismo conjunto de ejes
fig, axs = plt.subplots()
# path = '/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/' #path CONAE
path = '/media/maxpower/Mauro/SABIA-mar/20230918_NirSwir_INVAP/' #path CASA
file =path+'Darks/Temp2/dark_temp2_20230920105004.352.mat'

vmin = 60
vmax = 1000
df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
im = axs.imshow(df, cmap='viridis', vmin=vmin, vmax=vmax)
axs.set_title('T_int = {} ms'.format(mat73.loadmat(file)['salida']['tiemposInt']))

# Llamar a la función para diferentes valores de threshold
thresholds = [250, 400, 500,800]  # Puedes ajustar estos valores
colors = ['b', 'g', 'cyan','k']  # Puedes ajustar estos colores
off = 1
for threshold, color in zip(thresholds, colors):
    plot_hot_pixels(file, threshold, axs, color,off)
    off +=10
    if off >20:
        off = -10

plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3], markersize=10),],
           labels=['T{}'.format(thresholds[0]),
                   'T{}'.format(thresholds[1]),
                   'T{}'.format(thresholds[2]),'T{}'.format(thresholds[3])])
plt.show()

