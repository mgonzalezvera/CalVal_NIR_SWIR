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
# plot_only_one(file)

def cargar_matrices_conteo(nombre_archivo):
    """
    Carga las matrices de conteo desde un archivo .npy.

    :param nombre_archivo: Nombre del archivo .npy que contiene las matrices.
    :return: Lista de matrices de conteo.
    """
    matrices_conteo = np.load(nombre_archivo)
    return matrices_conteo
def contar_true_en_rectangulo(matriz, x1, x2, y1, y2):
    submatriz = matriz[y1:y2 + 1, x1:x2 + 1]
    return np.sum(submatriz)
def plot_bands(x1,x2,band,color,ax,y_min=-20,y_max=20):
    ax.plot([x1, x1], [y_min, y_max], label=band,color=color,linestyle='--')
    ax.plot([x2, x2], [y_min, y_max], color=color,linestyle='--')
def plot_rect_bands(x1,x2,y1,y2,band,color,ax):
    rect = patches.Rectangle((x1, y2), x2 - x1, y1 - y2, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x1 - 60, (y1 + y2) / 2, band, color=color, fontsize=12)

def compare_dead_pixels():
    matrices = []
    path_file = './arrays/*.npy'
    files = sorted(glob.glob(path_file))
    for file in files:
        matriz = cargar_matrices_conteo(file)
        matriz_booleana = matriz != 0
        matrices.append(matriz_booleana)
    pixeles_muertos_comunes = np.all(matrices, axis=0)
    # pixeles_muertos_comunes = np.all(matrices)

    conteo_true = np.sum(pixeles_muertos_comunes)
    # Crea una figura y un eje
    fig, ax = plt.subplots()

    # Configura los colores para plotear la matriz
    norm = plt.Normalize(0, 1)  # 0 para False (negro) y 1 para True (blanco)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['k', 'w'])
    # Plotea la matriz con colores blanco y negro
    ax.imshow(pixeles_muertos_comunes, cmap=cmap, norm=norm)

    # Encuentra las coordenadas de los píxeles con valores True
    coordenadas_true = np.argwhere(pixeles_muertos_comunes)

    # Marca los píxeles con valores True con círculos rojos
    pixel_count = 1
    for coord in coordenadas_true:
        y, x = coord
        ax.add_patch(plt.Circle((x, y), 2, color='red', fill=False))
        ax.annotate('{}'.format(pixel_count), (x, y), color='r', fontsize=25)
        pixel_count+=1
    porcentaje = conteo_true/pixeles_muertos_comunes.size*100
    # Muestra el recuento de valores True en la esquina superior derecha en rojo
    ax.text(0.9, 0.9, f'Dead Pixels: {conteo_true}\n {np.round(porcentaje,3)} percent of Total', color='red', transform=ax.transAxes, fontsize=25,
            horizontalalignment='right', verticalalignment='top')
    # Configura el aspecto de la gráfica
    ax.set_aspect('equal', adjustable='box')

    # Rectangulos
    bandas = {
        'B12': {'y1': 412, 'y2': 422, 'color': 'r'},
        'B10': {'y1': 451, 'y2': 459, 'color': 'b'},
        'B9': {'y1': 489, 'y2': 497, 'color': 'g'},
        'B11': {'y1': 526, 'y2': 535, 'color': 'y'},
        'B13': {'y1': 564, 'y2': 573, 'color': 'orange'},
        'B14': {'y1': 602, 'y2': 612, 'color': 'm'},
    }
    xlims = {'x1': 158, 'x2': 1122}
    for key in bandas.keys():
        plot_rect_bands(xlims['x1'] - 1, xlims['x2'] - 1, bandas[key]['y1'] - 1, bandas[key]['y2'] - 1, key,
                        bandas[key]['color'], ax)

    # Recorre los rectángulos y cuenta los valores True dentro de cada uno
    for key, rectangulo in bandas.items():
        x1, x2, y1, y2 = xlims['x1'] - 1, xlims['x2'] - 1, rectangulo['y1'] - 1, rectangulo['y2'] - 1
        conteo_true = contar_true_en_rectangulo(pixeles_muertos_comunes, x1, x2, y1, y2)

        # Dibuja el rectángulo en el gráfico
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=rectangulo['color'], facecolor='none')
        ax.add_patch(rect)

        # Anota el recuento de valores True al lado del rectángulo en rojo
        ax.text(x2 + 5, y2 + 5, f'True: {conteo_true}', color=rectangulo['color'], fontsize=15)

    # GUARDAR CAPA
    np.save('./arrays/dead_pixels_mask.npy', pixeles_muertos_comunes)    # Muestra la gráfica
    plt.show()

compare_dead_pixels()