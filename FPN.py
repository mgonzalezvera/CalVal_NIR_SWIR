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

 # Define las rutas de los archivos
file1 = path + 'Darks/Temp2/dark_temp2_20230920104852.959.mat'
file2 = path + 'Darks/Temp2/dark_temp2_20230920104856.050.mat'

# Carga las imágenes desde los archivos
df1 = np.mean(mat73.loadmat(file1)['salida']['imagen'], axis=2)
df2 = np.mean(mat73.loadmat(file2)['salida']['imagen'], axis=2)

# Extrae las dimensiones de las imágenes
alto, ancho = df1.shape

# Figura 1: Gráficos de Columnas apilados verticalmente
plt.figure(figsize=(8, 10))
plt.suptitle('Valores de los Píxeles por Columna')

plt.subplot(2, 1, 1)
for columna in range(ancho):
    plt.plot(range(alto), df1[:, columna], label=f'Columna {columna}')
plt.title('Imagen 1')
plt.xlabel('Fila')
plt.ylabel('Valor del Píxel')
plt.legend()

plt.subplot(2, 1, 2)
for columna in range(ancho):
    plt.plot(range(alto), df2[:, columna], label=f'Columna {columna}')
plt.title('Imagen 2')
plt.xlabel('Fila')
plt.ylabel('Valor del el Píxel')
plt.legend()

# Figura 2: Gráficos de Filas uno al lado del otro
plt.figure(figsize=(12, 5))
plt.suptitle('Valores de los Píxeles por Fila')

plt.subplot(1, 2, 1)
for fila in range(alto):
    plt.plot(range(ancho), df1[fila, :], label=f'Fila {fila}')
plt.title('Imagen 1')
plt.xlabel('Columna')
plt.ylabel('Valor del Píxel')
plt.legend()

plt.subplot(1, 2, 2)
for fila in range(alto):
    plt.plot(range(ancho), df2[fila, :], label=f'Fila {fila}')
plt.title('Imagen 2')
plt.xlabel('Columna')
plt.ylabel('Valor del Píxel')
plt.legend()

plt.show()


