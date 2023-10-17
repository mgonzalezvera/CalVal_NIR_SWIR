import mat73
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scienceplots

sns.light_palette("seagreen", as_cmap=True)
plt.style.use(['science'])
plt.rcParams['text.usetex'] = True
# --------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
def plot_rect_bands(x1,x2,y1,y2,band,color,ax):
    rect = patches.Rectangle((x1, y2), x2 - x1, y1 - y2, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x1 - 60, (y1 + y2) / 2, band, color=color, fontsize=12)
def plot_bands(x1,x2,band,color,ax,y_min=-20,y_max=20):
    ax.plot([x1, x1], [y_min, y_max], label=band,color=color,linestyle='--')
    ax.plot([x2, x2], [y_min, y_max], color=color,linestyle='--')
# --------------------------------------------------------------------------------------------------------------------
# Code Source
path = '/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/'
folders = [
    'Auxiliar',
    'Darks',
    'Rad',
    'Spectral',
    'Spectral',
    'Sphere_INVAP'
]

# --------------------------------------------------------------------------------------------------------------------
# DARKS
files_darks = sorted(glob.glob(path+'Darks/Temp1/*'))
data_dark = mat73.loadmat(files_darks[0])
# ..........................
# TIEMPOS DE INTEGRACION
# for file in files_darks:
#     dt = mat73.loadmat(file)['salida']
#     print(file)
#     print('T_integracion: ', dt['tiemposInt'])
#     print('-------------------------------------------------')

vmin = 60  # Valor mínimo personalizado
vmax = 180  # Valor máximo personalizado
cmap = plt.get_cmap('viridis')
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(3,3)
for i in range(0,9):
    row = i // 3
    col = i % 3
    df = np.mean(mat73.loadmat(files_darks[i])['salida']['imagen'],axis=2)
    im = axs[row,col].imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[row, col].set_title('T_int = {} ms'.format(mat73.loadmat(files_darks[i])['salida']['tiemposInt']))
cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('DN')


fig, axs = plt.subplots(3,3)
for i in range(0,9):
    row = i // 3
    col = i % 3
    df = np.mean(mat73.loadmat(files_darks[i+9])['salida']['imagen'],axis=2)
    im = axs[row,col].imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[row, col].set_title('T_int = {} ms'.format(mat73.loadmat(files_darks[i+9])['salida']['tiemposInt']))
cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('DN')

fig, axs = plt.subplots(3,3)
for i in range(0,9):
    row = i // 3
    col = i % 3
    df = np.mean(mat73.loadmat(files_darks[i+18])['salida']['imagen'],axis=2)
    im = axs[row,col].imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[row, col].set_title('T_int = {} ms'.format(mat73.loadmat(files_darks[i+18])['salida']['tiemposInt']))
cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('DN')

fig, axs = plt.subplots(3,3)
for i in range(0,9):
    row = i // 3
    col = i % 3
    df = np.mean(mat73.loadmat(files_darks[i+27])['salida']['imagen'],axis=2)
    im = axs[row,col].imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[row, col].set_title('T_int = {} ms'.format(mat73.loadmat(files_darks[i+27])['salida']['tiemposInt']))
cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('DN')

fig, axs = plt.subplots(3,3)
for i in range(0,9):
    row = i // 3
    col = i % 3
    df = np.mean(mat73.loadmat(files_darks[i+36])['salida']['imagen'],axis=2)
    im = axs[row,col].imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[row, col].set_title('T_int = {} ms'.format(mat73.loadmat(files_darks[i+36])['salida']['tiemposInt']))
cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('DN')
plt.show()
# ..........................

data_dark = data_dark['salida']
average_image_dark = np.mean(data_dark['imagen'], axis=2)
# --------------------------------------------------------------------------------------------------------------------

# RADIANCES
files_rad = sorted(glob.glob(path+'Rad/Temp1/Lcal/Images/*.mat'))
data_rad = mat73.loadmat(files_rad[0])
data_rad = data_rad['salida']

average_image = np.mean(data_rad['imagen'], axis=2)
x = np.linspace(0,np.shape(data_rad['imagen'])[1],np.shape(data_rad['imagen'])[1])
line = average_image[np.shape(data_rad['imagen'])[0]//2,:] - average_image_dark[np.shape(data_rad['imagen'])[0]//2,:]
# ----------------------------------------------------------------------------------------------------------------------
# FIGURES
# plot de radiancias
fig, ax = plt.subplots()
ax.plot(x, line)
# PLOTEO DE BANDAS
plot_bands(414,422,'B12(1044[nm])','r',ax)
plot_bands(454,458,'B10(765[nm])','b',ax)
plot_bands(492,496,'B09(750[nm])','g',ax)
plot_bands(529,535,'B11(865[nm])','y',ax)
plot_bands(566,572,'B13(1240[nm])','orange',ax)
plot_bands(605,610,'B14(1610[nm])','m',ax)
ax.legend()
# ----------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots()
vmin = 60  # Valor mínimo personalizado
vmax = 180  # Valor máximo personalizado
cmap = plt.get_cmap('viridis')
im = ax.imshow(average_image_dark, cmap=cmap, vmin=vmin, vmax=vmax)
cbar = plt.colorbar(im, ax=ax, orientation='vertical', ticks=[vmin, (vmin + vmax) / 2, vmax])
cbar.set_label('DN')
###################################################################################
# Rectangulos
bandas = {
    'B12': {'y1': 412,'y2':422,'color':'r' },
    'B10': {'y1': 451,'y2':459 ,'color':'b'},
    'B9': {'y1': 489,'y2':497 ,'color':'g'},
    'B11': {'y1': 526,'y2':535 ,'color':'y'},
    'B13': {'y1': 564,'y2':573 ,'color':'orange'},
    'B14': {'y1': 602,'y2':612 ,'color':'m'},
                }
xlims =  {'x1': 158,'x2':1122 }
for key in bandas.keys():
    plot_rect_bands(xlims['x1'],xlims['x2'],bandas[key]['y1'],bandas[key]['y2'],key,bandas[key]['color'],ax)
###################################################################################
# ----------------------------------------------------------------------------------------------------------------------
# HISTOGRAMA
fig, ax = plt.subplots()
plt.hist(average_image_dark,bins=30)
# ----------------------------------------------------------------------------------------------------------------------
plt.show()
# ----------------------------------------------------------------------------------------------------------------------


# fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# # Recorrer todas las capas y mostrarlas en los subplots
# for i in range(10):
#     row, col = divmod(i, 5)
#     ax = axes[row, col]
#     vmin = data_2['imagen'][:, :, i].min()
#     vmax=data_2['imagen'][:, :, i].max()
#     ax.imshow(data_2['imagen'][:, :, i], cmap='inferno', vmin=vmin, vmax=vmax)
#     ax.set_title(f'Capa {i}')
#     ax.axis('off')
#
# plt.tight_layout()
# plt.show()
#


# for key in data_1.keys():
#     print(key, data_1[key])
#
# for value in data_2['inforesult']:
#     print(value)