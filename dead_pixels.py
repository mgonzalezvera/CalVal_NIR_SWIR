import mat73
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scienceplots
from scipy.ndimage import sobel
from PIL import Image
import numpy as np
import csv
import plotly.express as px
import plotly.graph_objects as go
# --------------------------------------------------------------------------------------------------------------------
def guardar_matrices_conteo(matrices_conteo, nombre_archivo):
    """
    Guarda las matrices de conteo en un archivo .npy.

    :param matrices_conteo: Lista de matrices de conteo.
    :param nombre_archivo: Nombre del archivo .npy donde se guardarán las matrices.
    """
    np.save(nombre_archivo, matrices_conteo)
def cargar_matrices_conteo(nombre_archivo):
    """
    Carga las matrices de conteo desde un archivo .npy.

    :param nombre_archivo: Nombre del archivo .npy que contiene las matrices.
    :return: Lista de matrices de conteo.
    """
    matrices_conteo = np.load(nombre_archivo)
    return matrices_conteo


def encontrar_y_cortar_string(cadena):
    # Encuentra la frase "Temp1" en la cadena
    indice_temp1 = cadena.find("Temp")

    if indice_temp1 != -1:
        # Corta el string a partir de la posición de "Temp1"
        cadena_cortada = cadena[:indice_temp1+1 + len("Temp")+1]
        cadena_cortada = cadena_cortada.replace("/", "_")
        return cadena_cortada
    else:
        return None

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
# --------------------------------------------------------------------------------------------------------------------

sns.light_palette("seagreen", as_cmap=True)
plt.style.use(['science'])
plt.rcParams['text.usetex'] = True

path = '/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/' #path CONAE
# path = '/media/maxpower/Mauro/SABIA-mar/20230918_NirSwir_INVAP/' #path CASA

# --------------------------------------------------------------------------------------------------------------------
folders = [
    'Darks/Temp1/',
    'Darks/Temp2/',
    'Rad/Temp1/Lcal/Images/',
    'Rad/Temp1/Ltyp/Images/',
    'Rad/Temp2/Lmax/Images/',
    'Rad/Temp2/Ltyp/Images/'
]

def  dead_pixel_all():
    for folder in folders:
        fig, ax = plt.subplots()
        path_file = path + folder + '*.mat'
        files = sorted(glob.glob(path_file))
        df = np.mean(mat73.loadmat(files[0])['salida']['imagen'], axis=2)
        alto, ancho = df.shape
        matriz_de_recuento = np.zeros((alto, ancho), dtype=int)
        for file in files:
            df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
            pixeles_menores_a_25 = (df <= 22)
            matriz_de_recuento += pixeles_menores_a_25

        # Calcula el umbral como el 90% del total de imágenes
        umbral = 1 * len(files)

        # Encuentra las coordenadas de píxeles con conteo alto
        coordenadas_puntos_altos = np.argwhere(matriz_de_recuento >= umbral)

        # Crea un mapa de calor de la matriz de recuento
        im = ax.imshow(matriz_de_recuento, cmap='hot', interpolation='nearest')
        plt.colorbar(im)

        # Marca los píxeles con conteo alto con círculos rojos
        for coord in coordenadas_puntos_altos:
            y, x = coord
            ax.plot(x, y, 'ro', markersize=5)

            # Calcula el porcentaje
            porcentaje = (matriz_de_recuento[y, x] / len(files)) * 100

            # Dibuja el círculo alrededor del píxel
            circle = plt.Circle((x, y), 5, color='red', fill=False, lw=1)
            plt.gca().add_patch(circle)

            # Etiqueta el píxel con el porcentaje
            # plt.annotate(f'{porcentaje:.2f}%', (x, y), color='r', fontsize=10, ha='center', va='bottom')
            ax.annotate(f'{porcentaje:.2f}%', (x + 7, y + 7), color='r', fontsize=10, ha='center', va='bottom')

        ax.set_title("Dead Pixels - {}".format(folder))
        ax.axis('off')
        sufix = encontrar_y_cortar_string(folder)
        name_save = '/home/usuario/Documentos/MISION/CalVal/arrays/'+sufix
        if folder[:3] == 'Rad':
            name_save = name_save + '_' + folder[10:14]
        print('Saving array in: ',name_save)
        guardar_matrices_conteo(matriz_de_recuento,name_save)

    plt.show()

def compare_dead_pixels():
    matrices = []
    path_file = '/home/usuario/Documentos/MISION/CalVal/arrays/*.npy'
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
    for coord in coordenadas_true:
        y, x = coord
        ax.add_patch(plt.Circle((x, y), 2, color='red', fill=False))
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
    np.save('/home/usuario/Documentos/MISION/CalVal/arrays/dead_pixels_mask.npy', pixeles_muertos_comunes)    # Muestra la gráfica
    plt.show()

def find_coordinates_with_value_gt_zero(matrix_file, output_file):
    try:
        # Carga la matriz desde el archivo .npy
        matrix = np.load(matrix_file)

        # Encuentra las coordenadas donde el valor es mayor a cero
        rows, cols = np.where(matrix > 0)

        # Guarda las coordenadas en un archivo de salida
        coordinates = np.column_stack((rows, cols))
        np.savetxt(output_file, coordinates, delimiter=',', fmt='%d')
        print(f"Coordenadas donde el valor es mayor a cero guardadas en {output_file}")
    except Exception as e:
        print(f"Error: {e}")
def variacion_coordenadas(path,folder, coords):
    coordinates = []
    with open(coords, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            coordinates.append((int(row[0]), int(row[1])))

    path_file = path + folder + '*.mat'
    files = sorted(glob.glob(path_file))
    pixel_values = []
    timestamps = []
    mean_values = []
    std_values = []
    for file in files:
        data = mat73.loadmat(file)['salida']
        df = np.mean(data['imagen'], axis=2)
        mean_values.append(np.mean(df))
        std_values.append(np.std(df))
        pixel_values_in_image = [df[x, y] for x, y in coordinates]

        pixel_values.append(pixel_values_in_image)
        timestamps.append(data['tiemposInt'])
        # if data['tiemposInt'] >= 180:
        #     break
    pixel_values = np.array(pixel_values)
    timestamps = [float(timestamp) for timestamp in timestamps]
    print(timestamps)
    fig = None
    for i, pixel_value in enumerate(pixel_values.T):
        # plt.plot(timestamps, pixel_value, label=f'Pixel {i}')

        if fig is None:
            fig = px.line(x=timestamps, y=pixel_value, labels={'x': 'Tiempo', 'y': 'Valor del Píxel'},
                          title='Evolución de Valores de Píxeles')
        else:
            # Agrega las nuevas trazas a la figura existente
            fig.add_scatter(x=timestamps, y=pixel_value, name=f'Pixel {i}')

    fig.add_trace(go.Scatter(x=timestamps,
                             y=mean_values ,
                             name='Media ± Desviación Estándar'))

    fig.update_xaxes(tickangle=45)
    fig.show()
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# dead_pixel_all()
# compare_dead_pixels()
coords = '/home/usuario/Documentos/MISION/CalVal/arrays/coordenadas_dead_pixels.csv'
variacion_coordenadas(path,folders[0],coords)
# matrix_file = '/home/usuario/Documentos/MISION/CalVal/arrays/dead_pixels_mask.npy'  # Reemplaza con la ruta de tu archivo .npy
# output_file = '/home/usuario/Documentos/MISION/CalVal/arrays/coordenadas_dead_pixels.csv'  # Nombre del archivo de salida
# find_coordinates_with_value_gt_zero(matrix_file, output_file)
#
# folders = [
#     'Darks/Temp1/',
#     'Darks/Temp2/',
#     'Rad/Temp1/Lcal/Images/',
#     'Rad/Temp1/Ltyp/Images/',
#     'Rad/Temp2/Lmax/Images/',
#     'Rad/Temp2/Ltyp/Images/'
# ]
# path_file = '/home/usuario/Documentos/MISION/CalVal/arrays/*.npy'
# files = sorted(glob.glob(path_file))
# print(files)
# for i,file in enumerate(files):
#     fig, ax = plt.subplots()
#     matriz = cargar_matrices_conteo(file)
#     # Calcula el umbral como el 90% del total de imágenes
#     umbral = 1 * len(sorted(glob.glob(path+folders[i]+'*.mat')))
#
#     # Encuentra las coordenadas de píxeles con conteo alto
#     coordenadas_puntos_altos = np.argwhere(matriz >= umbral)
#     conteo = 0
#     for coord in coordenadas_puntos_altos:
#         conteo += 1
#         y, x = coord
#         ax.plot(x, y, 'ro', markersize=5)
#
#         # Calcula el porcentaje
#         porcentaje = (matriz[y, x] / len(sorted(glob.glob(path+folders[i]+'*.mat')))) * 100
#
#         # Dibuja el círculo alrededor del píxel
#         circle = plt.Circle((x, y), 5, color='red', fill=False, lw=1)
#         plt.gca().add_patch(circle)
#
#         # Etiqueta el píxel con el porcentaje
#         # plt.annotate(f'{porcentaje:.2f}%', (x, y), color='r', fontsize=10, ha='center', va='bottom')
#         ax.annotate(f'{porcentaje:.2f}%', (x + 7, y + 7), color='r', fontsize=10, ha='center', va='bottom')
#
#     im = ax.imshow(matriz, cmap='hot', interpolation='nearest')
#     ax.text(0.9, 0.9, f'Dead Pixels: {conteo}\n {np.round(porcentaje, 3)} percent of Total', color='red',
#             transform=ax.transAxes, fontsize=25,
#             horizontalalignment='right', verticalalignment='top')
#     ax.set_title("Dead Pixels - File: {}".format(file))
#     ax.axis('off')
#     plt.colorbar(im)
# plt.show()
# #
# #
