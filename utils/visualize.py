#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from random import randint
import numpy as np

class Visualization:
    # row = sagittal; col = coronal; z,slice = horizontal, axial
    def prepare_image(self, data, voxel_map):
        (col2coord, coord2col) = voxel_map
        vmin = -np.max(data) - 1
        fmri_image = np.full(coord2col.shape,0,dtype=float)
        row, col, axis = coord2col.shape
        for i in range(row):
            for j in range(col):
                for k in range(axis):
                    voxel = coord2col[i][j][k]
                    if 0 < voxel < len(data): fmri_image[i][j][k] = data[voxel]
        return fmri_image

    # def prepare_image(self, data, voxel_map):
    #     (col2coord, coord2col) = voxel_map
    #     fmri_image = np.zeros(coord2col.shape)
    #     for i in range(len(col2coord)):
    #         [x,y,z] = col2coord[i]
    #         voxel = coord2col[x][y][z]
    #         if voxel < len(data): fmri_image[x][y][z] = data[voxel]     #?????
    #     return fmri_image

    def show_slices(self, fmri_image, axis=2):
        shape = fmri_image.shape
        for i in range(shape[axis]):
            plt.imshow(fmri_image[:,:,i])
            plt.show()

    def plot_slices(self, fmri_image, filename=None, rows=4, cols=6):
        if not filename: filename = randint(0,10000000)
        fig=plt.figure(figsize=(50, 30))

        images = []
        for idx in range(fmri_image.shape[2]):
            fig.add_subplot(rows, cols, idx+1)
            img = plt.imshow(fmri_image[:,:,idx].T, cmap='jet')
            images += [img]
            # fig.colorbar(img)

        vmax = max(image.get_array().max() for image in images)
        vmin = min(image.get_array().min() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        for img in images:
            img.set_norm(norm)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
        fig.colorbar(images[0], cax=cbar_ax)

        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap()
                        or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())


        for im in images:
            im.callbacksSM.connect('changed', update)

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
        # fig.colorbar(img, cax=cbar_ax)
        plt.savefig("%s.pdf" % (filename))
        plt.show()

fmriviz = Visualization()

# %%
