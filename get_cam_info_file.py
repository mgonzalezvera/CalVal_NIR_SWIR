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

data_dark = mat73.loadmat(files_darks[0])
data_dark = data_dark['salida']
average_image_dark = np.mean(data_dark['imagen'], axis=2)
#
files_rad = sorted(glob.glob(path+'Rad/Temp1/Lcal/Images/*.mat'))
data_rad = mat73.loadmat(files_rad[0])
data_rad = data_rad['salida']

average_image = np.mean(data_rad['imagen'], axis=2)
x = np.linspace(0,np.shape(data_rad['imagen'])[1],np.shape(data_rad['imagen'])[1])
line = average_image[np.shape(data_rad['imagen'])[0]//2,:] - average_image_dark[np.shape(data_rad['imagen'])[0]//2,:]
plt.figure(1)
plt.plot(x, line)
# bandas
# B12(1044[nm])
plt.plot([414,414],[0,1],label='B12(1044[nm])')
plt.plot([422,422],[0,1])
# B10(765[nm])
plt.plot([454,454],[0,1],label='B10(765[nm])')
plt.plot([458,458],[0,1])
# B09(750[nm])
plt.plot([492,492],[0,1],label='B09(750[nm])')
plt.plot([496,496],[0,1])
# B11(865[nm])
plt.plot([529,529],[0,1],label='B11(865[nm])')
plt.plot([535,535],[0,1])
# B13(1240[nm])
plt.plot([566,566],[0,1],label='B13(1240[nm])')
plt.plot([572,572],[0,1])
# B14(1610[nm])
plt.plot([605,605],[0,1],label='B14(1610[nm])')
plt.plot([610,610],[0,1])
plt.legend()
plt.figure(2)
plt.imshow(average_image, cmap='inferno')
plt.show()

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