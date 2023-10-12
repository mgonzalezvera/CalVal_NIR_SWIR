import mat73
import glob
import matplotlib.pyplot as plt
import numpy as np

path = '/home/usuario/Documentos/MISION/CalVal/20230918_NirSwir_INVAP/'
folders = [
    'Auxiliar',
    'Darks',
    'Rad',
    'Spectral',
    'Spectral',
    'Sphere_INVAP'
]

files_darks = sorted(glob.glob(path+'Darks/Temp1/*'))

data_1 = mat73.loadmat(files_darks[0])
# data_2 = mat73.loadmat(files_darks[1])
data_1 = data_1['salida']
# data_2 = data_2['salida']
#
files_rad = sorted(glob.glob(path+'Rad/Temp1/Lcal/Images/*.mat'))
data_2 = mat73.loadmat(files_rad[0])
data_2 = data_2['salida']
#
average_image = np.mean(data_1['imagen'], axis=2)
plt.imshow(average_image, cmap='inferno')
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
# Recorrer todas las capas y mostrarlas en los subplots
for i in range(10):
    row, col = divmod(i, 5)
    ax = axes[row, col]
    vmin = data_2['imagen'][:, :, i].min()
    vmax=data_2['imagen'][:, :, i].max()
    ax.imshow(data_2['imagen'][:, :, i], cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_title(f'Capa {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()



# for key in data_1.keys():
#     print(key, data_1[key])

for value in data_2['inforesult']:
    print(value)