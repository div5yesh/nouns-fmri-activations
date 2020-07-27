#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from random import randint
import numpy as np

class Visualization:
    # row = sagittal; col = coronal; z,slice = horizontal, axial
    def prepare_image(self, data, voxel_map, fill=0):
        (col2coord, coord2col) = voxel_map
        fmri_image = np.full(coord2col.shape,fill,dtype=float)
        row, col, axis = coord2col.shape
        for i in range(row):
            for j in range(col):
                for k in range(axis):
                    voxel = coord2col[i][j][k] - 1
                    if 0 <= voxel < len(data): fmri_image[i][j][k] = data[voxel]
        return fmri_image

    # def prepare_image(self, data, voxel_map):
    #     (col2coord, coord2col) = voxel_map
    #     fmri_image = np.zeros(coord2col.shape)
    #     for i in range(len(col2coord)):
    #         [x,y,z] = col2coord[i]
    #         voxel = coord2col[x][y][z]
    #         if voxel < len(data): fmri_image[x][y][z] = data[voxel]     #?????
    #     return fmri_image

    def show_slices(self, fmri_image, vmin,vmax, filename, slice=-1):
        norm = colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
        if slice == -1:
            shape = fmri_image.shape
            for i in range(shape[2]):
                plt.imshow(fmri_image[:,:,i].T, cmap="jet", norm=norm)
                plt.savefig("%s.pdf" % (filename + str(i)), bbox_inches = 'tight', pad_inches=0)
                plt.show()
        else:
            plt.imshow(fmri_image[:,:,slice].T, cmap="jet", norm=norm)
            plt.savefig("%s.pdf" % (filename + str(i)), bbox_inches = 'tight', pad_inches=0)
            plt.show()

    def plot_slices(self, fmri_image, vmin, vmax, filename=None, size=(50,30), rows=4, cols=6, cmap="jet"):
        if not filename: filename = randint(0,10000000)
        fig=plt.figure(figsize=size)
        
        norm = colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)

        images = []
        for idx in range(fmri_image.shape[2]):
            fig.add_subplot(rows, cols, idx+1)
            img = plt.imshow(fmri_image[:,:,idx].T, cmap=cmap, norm=norm)
            images += [img]
            # fig.colorbar(img)

        # vmax = max(image.get_array().max() for image in images)
        # vmin = min(image.get_array().min() for image in images)

        # for img in images:
        #     img.set_norm(norm)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
        fig.colorbar(images[0], cax=cbar_ax)

        # def update(changed_image):
        #     for im in images:
        #         if (changed_image.get_cmap() != im.get_cmap()
        #                 or changed_image.get_clim() != im.get_clim()):
        #             im.set_cmap(changed_image.get_cmap())
        #             im.set_clim(changed_image.get_clim())

        # for im in images:
        #     im.callbacksSM.connect('changed', update)

        plt.savefig("%s.pdf" % (filename), bbox_inches = 'tight', pad_inches=0)
        plt.show()

fmriviz = Visualization()

# %%
