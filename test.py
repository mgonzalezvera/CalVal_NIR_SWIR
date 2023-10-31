# import numpy as np
# import matplotlib.pyplot as plt
#
# # Cargar la máscara desde un archivo CSV
# loaded_matrix = np.load('/home/usuario/Documentos/MISION/CalVal/arrays/dead_pixels_mask.npy')
#
# # Crear una figura y un eje
# fig, ax = plt.subplots()
#
# # Configurar colores (True: blanco, False: negro)
# cmap = plt.cm.gray  # Colores en escala de grises
#
# # Mostrar la máscara en el gráfico
# ax.imshow(loaded_matrix, cmap=cmap)
#
# # Configurar el aspecto de la gráfica
# ax.set_aspect('equal', adjustable='box')
# ax.set_xticks([])
# ax.set_yticks([])
#
# # Mostrar el gráfico
# plt.show()
import mat73
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scienceplots
from scipy.ndimage import sobel

sns.light_palette("seagreen", as_cmap=True)
plt.style.use(['science'])
plt.rcParams['text.usetex'] = True

# path = '/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/' #path CONAE
path = '/media/maxpower/Mauro/SABIA-mar/20230918_NirSwir_INVAP/' #path CASA

def plot_only_one(file,cuadro_1,cuadro_2):
    x1_1, x2_1, y1_1, y2_1 = cuadro_1
    x1_2, x2_2, y1_2, y2_2 = cuadro_2
    vmin = 130  # Valor mínimo personalizado
    vmax = 150  # Valor máximo personalizado
    cmap = plt.get_cmap('viridis')
    THRESHOLD = 22
    fig, axs = plt.subplots()
    df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
    subimagen_1 = df[y1_1:y2_1, x1_1:x2_1]
    subimagen_2 = df[y1_2:y2_2, x1_2:x2_2]
    media_cuadro_1 = np.mean(subimagen_1)
    media_cuadro_2 = np.mean(subimagen_2)
    print('Media cuadro arriba: ',media_cuadro_1)
    print('Media cuadro abajo: ',media_cuadro_2)
    im = axs.imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs.set_title('T_int = {} ms'.format(mat73.loadmat(file)['salida']['tiemposInt']))
    coords = np.argwhere(df <= THRESHOLD)
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]
    axs.plot(x_coords, y_coords, 'ro', markersize=2)
    # axs.annotate(f'Dead Pixels: {len(x_coords)}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
    #                        textcoords='offset points', color='r', fontsize=30, ha='right', va='top')
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cax)
    axs.set_title(file)
    cbar.set_label('DN')
    print('tiempo integrtacicon: ',mat73.loadmat(file)['salida']['tiemposInt'])
    # plt.show()


# file =path+'Rad/Temp2/Lmax/Images/rad_temp2_03000_20230920120552.565.mat'
# # file =path+'Rad/Temp2/Ltyp/Images/rad_temp2_00500_20230920141942.885.mat'
file1 =path+'Darks/Temp2/dark_temp2_20230920104843.324.mat'
file2 =path+'Darks/Temp2/dark_temp2_20230920105004.352.mat'
#

# # cuadros (x1,x2,y1,y2)
cuadro_1 = (254,454,762,904)
cuadro_2 = (254,454,937,1022)
# plot_only_one(file1,cuadro_1,cuadro_2)
plot_only_one(file2,cuadro_1,cuadro_2)
plt.show()
# file =path+'Darks/Temp2/dark_temp2_20230920105004.352.mat'
#
# # Supongamos que ya tienes la imagen promediada en escala de grises en la variable df
# df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Supongamos que ya tienes la imagen promediada en escala de grises en la variable df
#
# # Escalar la imagen a un rango válido para la conversión a BGR
# min_val = df.min()
# max_val = df.max()
# df_scaled = ((df - min_val) / (max_val - min_val) * 255).astype(np.uint8)
#
# # Aplica el operador Sobel para resaltar las diferencias verticales
# sobel = cv2.Sobel(df_scaled, cv2.CV_64F, 0, 1, ksize=5)
#
# # Aplica umbral para identificar las áreas de las líneas horizontales
# umbral = 5000  # Ajusta este valor según tus necesidades
# lineas = np.where(sobel > umbral)
#
# # Encuentra las coordenadas de inicio y final de las líneas
# coordenadas_inicio = np.min(lineas, axis=1)
# coordenadas_final = np.max(lineas, axis=1)
#
# # Crea una imagen BGR a partir de la imagen escalada
# imagen_color = cv2.cvtColor(df_scaled, cv2.COLOR_GRAY2BGR)
#
# # Dibuja las líneas detectadas en la imagen original
# for inicio, final in zip(coordenadas_inicio, coordenadas_final):
#     cv2.line(imagen_color, (0, inicio), (imagen_color.shape[1], inicio), (0, 0, 255), 2)
#
# # Muestra la imagen con las líneas resaltadas
# plt.imshow(imagen_color)
# plt.title('Líneas Horizontales Detectadas')
# plt.axis('off')
# plt.show()
