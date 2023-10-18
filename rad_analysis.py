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
path = '/media/maxpower/Mauro/SABIA-mar/20230918_NirSwir_INVAP/' #path CASA

files_rads = sorted(glob.glob(path+'Rad/Temp1/Lcal/Images/*.mat'))
contador = 0
# for i,file in enumerate(files_rads):
#     # df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
#     # df = np.mean(mat73.loadmat(file)['salida']['imagen'], axis=2)
#     print(file[-38:],'\t\tT_int = {} ms'.format(mat73.loadmat(file)['salida']['tiemposInt']),'Index {}'.format(i))
    # if df['salida']['tiemposInt'] >
vmin = 60  # Valor mínimo personalizado
vmax = 180  # Valor máximo personalizado
cmap = plt.get_cmap('viridis')
THRESHOLD = 30
aux = 94
fig, axs = plt.subplots(3, 3)
for i in range(0, 9):
    row = i // 3
    col = i % 3
    df = np.mean(mat73.loadmat(files_rads[i+aux])['salida']['imagen'], axis=2)
    im = axs[row, col].imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[row, col].set_title('T_int = {} ms'.format(mat73.loadmat(files_rads[i+aux])['salida']['tiemposInt']))
    coords = np.argwhere(df < THRESHOLD)
    x_coords = coords[:, 1]
    y_coords = coords[:, 0]
    axs[row, col].plot(x_coords, y_coords, 'ro', markersize=2)
    axs[row, col].annotate(f'Cold Points: {len(x_coords)}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                           textcoords='offset points', color='blue', fontsize=20, ha='right', va='top')

cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('DN')
plt.show()