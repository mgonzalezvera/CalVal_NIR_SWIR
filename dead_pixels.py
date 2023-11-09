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
from tqdm import tqdm

# --------------------------------------------------------------------------------------------------------------------
# Utility Functions
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
    # Encuentra la frase "Temp" en la cadena
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
def dividir_lista_creciente(lista):
    segmentos = []
    segmento_actual = [lista[0]]

    for i in range(1, len(lista)):
        if lista[i] > lista[i - 1]:
            segmento_actual.append(lista[i])
        else:
            segmentos.append(segmento_actual)
            segmento_actual = [lista[i]]

    segmentos.append(segmento_actual)

    return segmentos
def obtener_valores_y_vecinos(imagen, x, y):
    # Valores de los píxeles en las coordenadas dadas y sus vecinos
    valores = {
        'center':   imagen[y, x],
        'up':       imagen[y - 1, x],  # Píxel de arriba
        'down':     imagen[y + 1, x],  # Píxel de abajo
        'left':     imagen[y, x - 1],  # Píxel de la izquierda
        'right':    imagen[y, x + 1],  # Píxel de la derecha
        'upper_left': imagen[y - 1, x - 1],  # Esquina superior izquierda
        'upper_right': imagen[y - 1, x + 1], # Esquina superior derecha
        'lower_left': imagen[y + 1, x - 1],  # Esquina inferior izquierda
        'lower_right': imagen[y + 1, x + 1]  # Esquina inferior derecha
    }
    return valores
# --------------------------------------------------------------------------------------------------------------------
# GENERAL PARAMETERS
sns.light_palette("seagreen", as_cmap=True)
plt.style.use(['science'])
plt.rcParams['text.usetex'] = True
# --------------------------------------------------------------------------------------------------------------------
def  dead_pixel_all(folders):
    """
    Función que analiza imágenes para identificar píxeles defectuosos.
    Calcula matrices de recuento basadas en un umbral (THRESHOLD), contando la cantidad de píxeles que tienen valores
    por debajo de este umbral para cada tiempo de integración en las subcarpetas proporcionadas. Guarda las matrices
    obtenidas en la subcarpeta "./arrays/" con extension ".npy".
    Posteriormente, genera gráficos que destacan los píxeles que cumplen con esta condición.

    :param folders: Lista de carpetas que contienen los archivos de imágenes a analizar.
    :return: No devuelve ningún valor, pero genera y muestra
    gráficos que proporcionan información sobre los píxeles defectuosos.
    """
    for folder in folders:
        fig, ax = plt.subplots()
        path_file = path + folder + '*.mat'
        files = sorted(glob.glob(path_file))
        df = np.mean(mat73.loadmat(files[0])['salida']['imagen'], axis=2)
        alto, ancho = df.shape
        THRESHOLD = 22 #umbral de deteccion de pixel "malo"
        matriz_de_recuento = np.zeros((alto, ancho), dtype=int)
        for file in files:
            df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
            pixeles_menores_a_25 = (df <= THRESHOLD)
            matriz_de_recuento += pixeles_menores_a_25

        # Calcula el umbral como el 90% del total de imágenes
        umbral = 0.1 * len(files)

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
        name_save = './arrays/'+sufix
        if folder[:3] == 'Rad':
            name_save = name_save + '_' + folder[10:14]
        print('Saving array in: ',name_save)
        guardar_matrices_conteo(matriz_de_recuento,name_save)

    plt.show()

def compare_dead_pixels():
    """
    Función que compara y visualiza píxeles muertos comunes en varias matrices de recuento.
    Carga matrices de recuento creadas por la funcion "dead_pixels" desde archivos '.npy' en la carpeta './arrays'.
    Luego, encuentra píxeles muertos comunes entre las matrices y genera un gráfico que destaca estos píxeles comunes.

    :return: No devuelve ningún valor, pero genera y muestra un gráfico que resalta los píxeles muertos comunes.
    """
    matrices = []
    path_file = './arrays/*.npy'
    files = sorted(glob.glob(path_file))
    for file in files:
        if file != 'dead_pixels_mask.npy':
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
    np.save('./arrays/dead_pixels_mask.npy', pixeles_muertos_comunes)    # Muestra la gráfica
    plt.show()

def find_coordinates_with_value_gt_zero(matrix_file, output_file):
    """
    Función que encuentra las coordenadas donde los valores de una matriz son mayores a cero.
    Carga la matriz desde un archivo '.npy' especificado en 'matrix_file', identifica las coordenadas con
    valores mayores a cero y guarda estas coordenadas en un archivo de salida especificado en 'output_file'.

    :param matrix_file: Ruta del archivo '.npy' que contiene la matriz.
    :param output_file: Ruta del archivo de salida para guardar las coordenadas.
    :return: No devuelve ningún valor, pero guarda las coordenadas en un archivo de salida.
    """
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
    """
    Función que analiza la variación de los valores de píxeles en el tiempo.
    Carga las coordenadas desde un archivo CSV especificado en 'coords' y los datos de imágenes desde archivos '.mat'
    en la carpeta proporcionada. Luego, realiza cálculos estadísticos sobre regiones específicas de las imágenes (media
    y desvio en regiones donde se observa un comportamiento normal de los pixels) y muestra la evolución temporal de
    los valores de los píxeles seleccionados en "coords", junto con la media y la desviación estándar.

    :param path: Ruta de la carpeta donde se encuentran los archivos de imágenes '.mat'.
    :param folder: Nombre de la subcarpeta dentro de 'path' que se está analizando.
    :param coords: Ruta del archivo CSV que contiene las coordenadas de los píxeles a analizar.
    :return: No devuelve ningún valor, pero muestra un gráfico que ilustra la evolución temporal de los valores
    de píxeles, la media y la desviación estándar.
    """
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
    # ----------------------------------------------------------
    # cuadros por fuera de los filtros y de superficies de referencia
    cuadro_1 = (75, 1200, 66, 361)
    cuadro_2 = (26, 1256, 663, 955)
    x1_1, x2_1, y1_1, y2_1 = cuadro_1
    x1_2, x2_2, y1_2, y2_2 = cuadro_2

    # ----------------------------------------------------------
    for file in files:
        data = mat73.loadmat(file)['salida']
        df = np.mean(data['imagen'], axis=2)
        subimagen_1 = df[y1_1:y2_1, x1_1:x2_1]
        subimagen_2 = df[y1_2:y2_2, x1_2:x2_2]
        media_cuadro_1 = np.mean(subimagen_1)
        media_cuadro_2 = np.mean(subimagen_2)
        mean_values.append(np.mean([media_cuadro_1,media_cuadro_2]))
        std_values.append(np.std([media_cuadro_1,media_cuadro_2]))
        pixel_values_in_image = [df[x, y] for x, y in coordinates]
        pixel_values.append(pixel_values_in_image)
        timestamps.append(data['tiemposInt'])
        # if data['tiemposInt'] >= 180:
        #     break
    pixel_values = np.array(pixel_values)
    # print(pixel_values.T.shape)
    # print(pixel_values.T)
    timestamps = [float(timestamp) for timestamp in timestamps]
    timestamps = dividir_lista_creciente(timestamps)
    # print(timestamps)
    fig = None
    # print(timestamps[0])
    # print(len(timestamps[0]))
    # print('Mean')
    # print(mean_values)
    for i, pixel_value in enumerate(pixel_values[:len(timestamps[0])].T):
        plt.plot(timestamps[0], pixel_value, label=f'Pixel {i}')
        # plt.show()

        # if fig is None:
        #     fig = px.line(x=timestamps[0], y=pixel_value, labels={'x': 'Tiempo', 'y': 'Valor del Píxel'},
        #                   title='Evolución de Valores de Píxeles - {}'.format(folder))
        # else:
        #     # Agrega las nuevas trazas a la figura existente
        #     fig.add_scatter(x=timestamps[0], y=pixel_value, name=f'Pixel {i}')

    # fig.add_trace(go.Scatter(x=timestamps[0],
    #                          y=mean_values ,
    #                          name='Media ± Desviación Estándar'))

    # fig.add_trace(go.Scatter(x=timestamps[0], y=mean_values, name='Media'))
    #
    # fig.add_trace(go.Scatter(x=timestamps[0], y=np.array(mean_values) + std_values, fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), name=f'Dispersión {i}'))
    # fig.add_trace(go.Scatter(x=timestamps[0], y=np.array(mean_values) - std_values, fill='tonexty', mode='none', fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), name=f'Dispersión {i}'))
    #
    # fig.update_xaxes(tickangle=45)
    # fig.show()
    mean_values = np.array(mean_values)[:len(timestamps[0])]
    std_values = np.array(std_values)[:len(timestamps[0])]

    plt.plot(timestamps[0], mean_values[:len(timestamps[0])], label='Mean')


    plt.fill_between(timestamps[0],mean_values - std_values, mean_values + std_values, alpha=0.2, label='Standard Deviation')
    plt.title(folder,fontsize=20)
    plt.xlabel('Integration time [ms]',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('DN',fontsize=20)
    plt.legend(ncol=2,frameon=True,loc='upper right')
    plt.grid()
    plt.show()

def var_pixels_adyac(folder):
    """
    Función que analiza la variación de los valores de píxeles en una posición específica y sus vecinos a lo largo
    del tiempo. Lee archivos de imágenes '.mat' en la carpeta proporcionada y muestra un gráfico que ilustra la
    evolución temporal de los valores de píxeles en la posición central y sus vecinos.

    :param folder: Nombre de la subcarpeta que contiene los archivos de imágenes '.mat' a analizar.
    :return: No devuelve ningún valor, pero muestra un gráfico que representa la variación temporal de los valores de píxeles en una posición específica y sus vecinos.
    """
    path_file = path + folder + '*.mat'
    files = sorted(glob.glob(path_file))
    # Archivo de coordenadas
    archivo_coord = './arrays/coordenadas_dead_pixels.csv'
    # Leer las coordenadas desde el archivo
    coordenadas = np.loadtxt(archivo_coord, delimiter=',', dtype=int)
    for i, (y, x) in enumerate(coordenadas):
        print('----------------------------')
        print('Reading point {} of {}'.format(i,len(coordenadas)))
        plt.figure(figsize=(8, 6))
        plt.title('Pixel: {}'.format(i))
        plt.xlabel('Integration time [ms]',fontsize=20)
        plt.ylabel('DN',fontsize=20)
        values = {
            'center':[],
            'up': [],
            'down': [],
            'left': [],
            'right': [],
            'upper_left': [],
            'upper_right': [],
            'lower_left': [],
            'lower_right': [],
            'int_time': []
        }
        total_archivos = len(files)
        for file in tqdm(range(1, total_archivos + 1), desc='Leyendo archivos de carpeta: '.format(folder)):
            # print(file)
        # for number,file in enumerate(files):
            # print('Reding file: {} of {}'.format(number,len(files)), end='\r')
            # data = mat73.loadmat(file)['salida']
            data = mat73.loadmat(files[file-1])['salida']
            int_time = data['tiemposInt']
            df = np.mean(data['imagen'], axis=2)
            valores = obtener_valores_y_vecinos(df, x, y)
            for key in values.keys():
                if key != 'int_time':
                    values[key].append(valores[key])
            values['int_time'].append(int_time)
        for key in values.keys():
            if key != 'int_time':
                plt.plot(values['int_time'],values[key],label=key)

        plt.legend(loc='upper right',fontsize=20,frameon=True)

    plt.show()

def plot_only_one(file):
    """
    Función que genera un gráfico para visualizar la imagen promedio y la ubicación de píxeles muertos en una
    sola imagen. Carga la imagen desde un archivo '.mat' especificado en 'file', y superpone los píxeles muertos
    utilizando las coordenadas del archivo 'coordenadas_dead_pixels.csv'.

    :param file: Ruta del archivo '.mat' que contiene la imagen a visualizar.
    :return: No devuelve ningún valor, pero muestra un gráfico que representa la imagen promedio y la ubicación de píxeles muertos.
    """
    vmin = 60  # Valor mínimo personalizado
    vmax = 500  # Valor máximo personalizado
    cmap = plt.get_cmap('viridis')
    fig, axs = plt.subplots()
    df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
    im = axs.imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs.set_title('T_int = {} ms'.format(mat73.loadmat(file)['salida']['tiemposInt']))

    archivo_coord = './arrays/coordenadas_dead_pixels.csv'
    coordenadas = np.loadtxt(archivo_coord, delimiter=',', dtype=int)
    for i, (y, x) in enumerate(coordenadas):
        axs.plot(x, y, 'ro', markersize=2)
        axs.annotate('{}'.format(i+1), (x, y), color='r', fontsize=25)

    # axs.plot(x_coords, y_coords, 'ro', markersize=2)
    # axs.annotate(f'Dead Pixels: {len(x_coords)}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
    #                        textcoords='offset points', color='r', fontsize=30, ha='right', va='top')
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cax)
    axs.set_title(file)
    cbar.set_label('DN')
    print('tiempo integrtacicon: ',mat73.loadmat(file)['salida']['tiemposInt'])
    # plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# SOURCE CODE

# path = '/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/' #path CONAE
path = '/media/maxpower/Mauro/SABIA-mar/20230918_NirSwir_INVAP/' #path CASA

folders = [
    'Darks/Temp1/',
    'Darks/Temp2/',
    'Rad/Temp1/Lcal/Images/',
    'Rad/Temp1/Ltyp/Images/',
    'Rad/Temp2/Lmax/Images/',
    'Rad/Temp2/Ltyp/Images/'
]

# folders = [
#     'Darks/Temp2/',
# ]

# dead_pixel_all(folders)

# compare_dead_pixels()



# matrix_file = './arrays/dead_pixels_mask.npy'  # Reemplaza con la ruta de tu archivo .npy
# output_file = './arrays/coordenadas_dead_pixels.csv'  # Nombre del archivo de salida
# find_coordinates_with_value_gt_zero(matrix_file, output_file)

#
# folders = [
#     'Darks/Temp2/',
#     'Rad/Temp1/Ltyp/Images/'
# ]
# coords = './arrays/coordenadas_dead_pixels.csv'
# for i,folder in enumerate(folders):
#     variacion_coordenadas(path,folders[i],coords)
# # #
# # variacion_coordenadas(path,folders[1],coords)
#
var_pixels_adyac(folders[0])
# file2 =path+'Darks/Temp2/dark_temp2_20230920105004.352.mat'
# plot_only_one(file2)
# plt.show()
# # folders = [
# #     'Darks/Temp1/',
# #     'Darks/Temp2/',
# #     'Rad/Temp1/Lcal/Images/',
# #     'Rad/Temp1/Ltyp/Images/',
# #     'Rad/Temp2/Lmax/Images/',
# #     'Rad/Temp2/Ltyp/Images/'
# # ]
# # path_file = '/home/usuario/Documentos/MISION/CalVal/arrays/*.npy'
# # files = sorted(glob.glob(path_file))
# # print(files)
# # for i,file in enumerate(files):
# #     fig, ax = plt.subplots()
# #     matriz = cargar_matrices_conteo(file)
# #     # Calcula el umbral como el 90% del total de imágenes
# #     umbral = 1 * len(sorted(glob.glob(path+folders[i]+'*.mat')))
# #
# #     # Encuentra las coordenadas de píxeles con conteo alto
# #     coordenadas_puntos_altos = np.argwhere(matriz >= umbral)
# #     conteo = 0
# #     for coord in coordenadas_puntos_altos:
# #         conteo += 1
# #         y, x = coord
# #         ax.plot(x, y, 'ro', markersize=5)
# #
# #         # Calcula el porcentaje
# #         porcentaje = (matriz[y, x] / len(sorted(glob.glob(path+folders[i]+'*.mat')))) * 100
# #
# #         # Dibuja el círculo alrededor del píxel
# #         circle = plt.Circle((x, y), 5, color='red', fill=False, lw=1)
# #         plt.gca().add_patch(circle)
# #
# #         # Etiqueta el píxel con el porcentaje
# #         # plt.annotate(f'{porcentaje:.2f}%', (x, y), color='r', fontsize=10, ha='center', va='bottom')
# #         ax.annotate(f'{porcentaje:.2f}%', (x + 7, y + 7), color='r', fontsize=10, ha='center', va='bottom')
# #
# #     im = ax.imshow(matriz, cmap='hot', interpolation='nearest')
# #     ax.text(0.9, 0.9, f'Dead Pixels: {conteo}\n {np.round(porcentaje, 3)} percent of Total', color='red',
# #             transform=ax.transAxes, fontsize=25,
# #             horizontalalignment='right', verticalalignment='top')
# #     ax.set_title("Dead Pixels - File: {}".format(file))
# #     ax.axis('off')
# #     plt.colorbar(im)
# # plt.show()
# # #
# # #