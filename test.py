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

path = '/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/' #path CONAE
# path = '/media/maxpower/Mauro/SABIA-mar/20230918_NirSwir_INVAP/' #path CASA

def plot_only_one(file):
    vmin = 60  # Valor mínimo personalizado
    vmax = 500  # Valor máximo personalizado
    cmap = plt.get_cmap('viridis')
    THRESHOLD = 22
    fig, axs = plt.subplots()
    df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
    im = axs.imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs.set_title('T_int = {} ms'.format(mat73.loadmat(file)['salida']['tiemposInt']))
    coords = np.argwhere(df < THRESHOLD)
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]
    axs.plot(x_coords, y_coords, 'ro', markersize=2)
    axs.annotate(f'Cold Points: {len(x_coords)}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                           textcoords='offset points', color='blue', fontsize=20, ha='right', va='top')
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('DN')
    plt.show()
file ='/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/Rad/Temp2/Lmax/Images/rad_temp2_07000_20230920114535.418.mat'
plot_only_one(file)
