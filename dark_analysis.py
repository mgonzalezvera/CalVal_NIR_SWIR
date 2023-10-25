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

# --------------------------------------------------------------------------------------------------------------------
# DARKS
files_darks = sorted(glob.glob(path+'Darks/Temp1/*'))
files_darks = sorted(glob.glob(path+'Rad/Temp2/Ltyp/Images/*'))
data_dark = mat73.loadmat(files_darks[0])
# ..........................
# TIEMPOS DE INTEGRACION
# for file in files_darks:
#     dt = mat73.loadmat(file)['salida']
#     print(file)
#     print('T_integracion: ', dt['tiemposInt'])
#     print('-------------------------------------------------')
# ----------------------------------------------------------------------------------------------------------------------
# OUTLIERS ANALYSIS
def outlier_analysis():
    vmin = 60  # Valor mínimo personalizado
    vmax = 180  # Valor máximo personalizado
    cmap = plt.get_cmap('viridis')
    UMBRAL = True
    THRESHOLD = 1000
    fig, axs = plt.subplots(3,3)
    for i in range(0,9):
        row = i // 3
        col = i % 3
        df = np.mean(mat73.loadmat(files_darks[i])['salida']['imagen'],axis=2)
        im = axs[row,col].imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[row, col].set_title('T_int = {} ms'.format(mat73.loadmat(files_darks[i])['salida']['tiemposInt']))
        if UMBRAL:
            cont_outlier = 0
            edges = sobel(df)
            threshold = THRESHOLD  # Ajusta este valor según tus necesidades
            outliers = edges > threshold
            for y, x in zip(*np.where(outliers)):
                axs[row, col].add_patch(plt.Circle((x, y), radius=5, color='red', fill=False))
                cont_outlier +=1
            axs[row, col].annotate(f'Outliers: {cont_outlier}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                                   textcoords='offset points', color='k', fontsize=20, ha='right', va='top')

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
        if UMBRAL:
            cont_outlier = 0
            edges = sobel(df)
            threshold = THRESHOLD  # Ajusta este valor según tus necesidades
            outliers = edges > threshold
            for y, x in zip(*np.where(outliers)):
                axs[row, col].add_patch(plt.Circle((x, y), radius=5, color='red', fill=False))
                cont_outlier += 1
            axs[row, col].annotate(f'Outliers: {cont_outlier}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                                   textcoords='offset points', color='k', fontsize=20, ha='right', va='top')
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
        if UMBRAL:
            cont_outlier = 0
            edges = sobel(df)
            threshold = THRESHOLD  # Ajusta este valor según tus necesidades
            outliers = edges > threshold
            for y, x in zip(*np.where(outliers)):
                axs[row, col].add_patch(plt.Circle((x, y), radius=5, color='red', fill=False))
                cont_outlier += 1
            axs[row, col].annotate(f'Outliers: {cont_outlier}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                                   textcoords='offset points', color='k', fontsize=20, ha='right', va='top')
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
        if UMBRAL:
            cont_outlier = 0
            edges = sobel(df)
            threshold = THRESHOLD  # Ajusta este valor según tus necesidades
            outliers = edges > threshold
            for y, x in zip(*np.where(outliers)):
                axs[row, col].add_patch(plt.Circle((x, y), radius=5, color='red', fill=False))
                cont_outlier += 1
            axs[row, col].annotate(f'Outliers: {cont_outlier}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                                   textcoords='offset points', color='k', fontsize=20, ha='right', va='top')



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
        if UMBRAL:
            cont_outlier = 0
            edges = sobel(df)
            threshold = THRESHOLD  # Ajusta este valor según tus necesidades
            outliers = edges > threshold
            for y, x in zip(*np.where(outliers)):
                axs[row, col].add_patch(plt.Circle((x, y), radius=5, color='red', fill=False))
                cont_outlier += 1
            axs[row, col].annotate(f'Outliers: {cont_outlier}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                                   textcoords='offset points', color='k', fontsize=20, ha='right', va='top')
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('DN')
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
def cold_points():
    vmin = 60  # Valor mínimo personalizado
    vmax = 180  # Valor máximo personalizado
    cmap = plt.get_cmap('viridis')
    THRESHOLD = 22
    fig, axs = plt.subplots(3,3)
    for i in range(0,9):
        row = i // 3
        col = i % 3
        df = np.mean(mat73.loadmat(files_darks[i])['salida']['imagen'],axis=2)
        im = axs[row,col].imshow(df, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[row, col].set_title('T_int = {} ms'.format(mat73.loadmat(files_darks[i])['salida']['tiemposInt']))
        coords = np.argwhere(df < THRESHOLD)
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        axs[row, col].plot(x_coords, y_coords, 'ro', markersize=2)
        axs[row, col].annotate(f'Cold Points: {len(x_coords)}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                               textcoords='offset points', color='blue', fontsize=20, ha='right', va='top')

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
        coords = np.argwhere(df < THRESHOLD)
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        axs[row, col].plot(x_coords, y_coords, 'ro', markersize=2)
        axs[row, col].annotate(f'Cold Points: {len(x_coords)}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                               textcoords='offset points', color='blue', fontsize=20, ha='right', va='top')
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
        coords = np.argwhere(df < THRESHOLD)
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        axs[row, col].plot(x_coords, y_coords, 'ro', markersize=2)
        axs[row, col].annotate(f'Cold Points: {len(x_coords)}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                               textcoords='offset points', color='blue', fontsize=20, ha='right', va='top')
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
        coords = np.argwhere(df < THRESHOLD)
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        axs[row, col].plot(x_coords, y_coords, 'ro', markersize=2)
        axs[row, col].annotate(f'Cold Points: {len(x_coords)}', xy=(1, 1), xycoords='axes fraction', xytext=(-10, -10),
                               textcoords='offset points', color='blue', fontsize=20, ha='right', va='top')
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
# ----------------------------------------------------------------------------------------------------------------------
cold_points()
# outlier_analysis()

