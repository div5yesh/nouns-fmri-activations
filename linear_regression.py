# %%
from itertools import groupby, combinations
import numpy as np
import scipy.io, pickle, os
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression

from utils.visualize import fmriviz
from utils.preprocess import dataloader, preprocess, postprocess

# %%
def train(features, trial_map, data_flat, idx=1):
    nouns = set(trial_map.keys())
    test_combinations = combinations(nouns, 2)
    iteration = 0
    predictions = []
    true_images = []
    labels = []

    # model
    reg_model = LinearRegression()

    for test_nouns in test_combinations:
        train_nouns = nouns - set(test_nouns)

        # prepare dataset
        train_images, train_features = preprocess.prepare_data(features, trial_map, data_flat, train_nouns)
        test_images, test_features = preprocess.prepare_data(features, trial_map, data_flat, test_nouns)

        # training
        reg_model.fit(train_features, train_images)
        pred_images = reg_model.predict(test_features)

        predictions += [pred_images]
        true_images += [test_images]
        labels += [test_nouns]

        iteration += 1
        if iteration % 100 == 0:
            print("Training Combination: %d/%d" % (iteration, 1770))
            # break

    return np.array(predictions), np.array(true_images), labels

def test(snr, predictions, true_images):
    arr_similarity = []
    for i in range(len(predictions)):
        similarity = postprocess.evaluate(snr, predictions[i], true_images[i],500)
        arr_similarity += [similarity]

    # accuracy = sum(arr_similarity * 1)/len(arr_similarity)
    # print('Accuracy: %f' % (accuracy))
    return arr_similarity

# %% ---------------------- train-------------------------------------------
all_pairs = []
for i in range(1,2):
    participant = i
    samples = dataloader.data[participant].samples
    voxel_map = dataloader.data[participant].voxel_map
    trial_map = dataloader.data[participant].trial_map
    features = dataloader.features

    print("Participant: %d" % (participant))
    predictions, true_images, labels = train(features, trial_map, samples)
    all_pairs += [[predictions,true_images, labels]]

# %% ---------------------- test-------------------------------------------
for i in range(1,2):
    participant = i
    samples = dataloader.data[participant].samples
    voxel_map = dataloader.data[participant].voxel_map
    trial_map = dataloader.data[participant].trial_map

    print("Participant: %d" % (participant))
    # snr = preprocess.get_snr(participant, samples, trial_map)
    grp_samples = preprocess.get_grouped_samples(trial_map, samples)
    snr = postprocess.correlation2(grp_samples.T)
    
    vmin = np.min(snr)
    vmax = np.max(snr)

    snr_img = fmriviz.prepare_image(snr, voxel_map, fill=vmin)
    fmriviz.plot_slices(snr_img, vmin, vmax, "PCR_P%d" % (participant))

    predictions = all_pairs[participant-1][0]
    true_images = all_pairs[participant-1][1]
    sim_mat = test(snr, predictions, true_images)

#%% ---------------------------------------------------------------------
train_vecs, embeds = preprocess.prepare_data(features,trial_map,samples,list(trial_map.keys()))
# trainX = prepare_images(train_vecs,voxel_map)

lblmap ={
    "celery": 17,
    "telephone": 54,
    "bottle": 12,
    "cup": 26,
    "corn": 24,
    "table": 53
}

for i in range(5):
    p1_gen = all_pairs[0][0][i,1]
    # p1_real = all_pairs[0][1][i,1]
    lbl = all_pairs[0][2][i][1]
    idx = lblmap[lbl]

    p1_real = train_vecs[idx]

    vmin = np.min(p1_real)
    vmax = np.max(p1_real)

    sample_image = fmriviz.prepare_image(p1_real, voxel_map, fill=vmin)
    fmriviz.plot_slices(sample_image,vmin,vmax, filename="LR_" + lbl + "_real")

    sample_image = fmriviz.prepare_image(p1_gen, voxel_map, fill=vmin)
    fmriviz.plot_slices(sample_image,vmin,vmax, filename="LR_" + lbl + "_gen")

   
# %%
top500 = postprocess.get_top_voxels(snr,500)
binary = np.full(snr.shape, -0.2)
binary[top500] = 1
snr_binimg = fmriviz.prepare_image(binary, voxel_map, -1)
fmriviz.plot_slices(snr_binimg, -1, 1, "PCR_bin_P%d" % (participant), cmap="gray_r")

# %%
top500 = postprocess.get_top_voxels(snr,1000)
binary = np.full(snr.shape, 0.2)
binary[top500] = 1
snr_binimg = fmriviz.prepare_image(binary, voxel_map)
fmriviz.plot_slices(snr_binimg, "SNR_bin1k_P%d" % (participant), cmap="gray_r")

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

participant = 1
samples = dataloader.data[participant].samples
voxel_map = dataloader.data[participant].voxel_map
trial_map = dataloader.data[participant].trial_map
features = dataloader.features

vmin = np.min(samples[0])
vmax = np.max(samples[0])

sample_image = fmriviz.prepare_image(samples[0], voxel_map, fill=vmin)
# pickle.dump(sample_image, open("LR_"+lbl+"_real", "wb"))

norm = colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
plt.axis('off')
plt.imshow(sample_image[:,21,:].T, cmap="jet", norm=norm)
plt.savefig("%s.pdf" % ("coronal21"), bbox_inches = 'tight', pad_inches=0)
plt.show()

# %%
