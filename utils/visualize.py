#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
class Visualization:
    # row = sagittal; col = coronal; z,slice = horizontal, axial
    # def prepare_image(self, data, voxel_map):
    #     fmri_image = np.zeros(voxel_map.shape)
    #     row, col, axis = voxel_map.shape
    #     for i in range(row):
    #         for j in range(col):
    #             for k in range(axis):
    #                 voxel = voxel_map[i][j][k]
    #                 fmri_image[i][j][k] = data[voxel - 1]
    #     return fmri_image

    def prepare_image(self, data, voxel_map):
        (col2coord, coord2col) = voxel_map
        fmri_image = np.zeros(coord2col.shape)
        for i in range(len(col2coord)):
            [x,y,z] = col2coord[i]
            voxel = coord2col[x][y][z]
            fmri_image[x][y][z] = data[voxel-1]     #?????
        return fmri_image

    def show_slices(self, fmri_image, axis=2):
        shape = fmri_image.shape
        for i in range(shape[axis]):
            plt.imshow(fmri_image[:,:,i])
            plt.show()

    def trim_axs(self, axs, N):
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    def plot_slices(self, fmri_image, rows=4, cols=6):
        idx = 0
        _, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(50,30), constrained_layout=True)
        axs = self.trim_axs(axs, fmri_image.shape[2])
        for ax in axs:
            ax.imshow(fmri_image[:,:,idx])
            idx += 1
        plt.show()

fmriviz = Visualization()

# %%
