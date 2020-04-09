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

        iteration += 1
        if iteration % 100 == 0:
            print("Training Combination: %d/%d" % (iteration, 1770))
            # break

    return np.array(predictions), np.array(true_images)

def test(snr, predictions, true_images):
    arr_similarity = []
    for i in range(len(predictions)):
        similarity = postprocess.evaluate(snr, predictions[i], true_images[i])
        arr_similarity += [similarity]

    accuracy = sum(arr_similarity * 1)/len(arr_similarity)
    print('Accuracy: %f' % (accuracy))

# %% ---------------------- train-------------------------------------------
all_pairs = []
for i in range(1,10):
    participant = i
    samples = dataloader.data[participant].samples
    voxel_map = dataloader.data[participant].voxel_map
    trial_map = dataloader.data[participant].trial_map
    features = dataloader.features

    print("Participant: %d" % (participant))
    predictions, true_images = train(features, trial_map, samples)
    all_pairs += [[predictions,true_images]]

# %% ---------------------- test-------------------------------------------
for i in range(1,10):
    participant = i
    samples = dataloader.data[participant].samples
    voxel_map = dataloader.data[participant].voxel_map
    trial_map = dataloader.data[participant].trial_map

    print("Participant: %d" % (participant))
    snr = preprocess.get_snr(participant, samples, trial_map)
    print(snr.shape)
    # snr_img = fmriviz.prepare_image(snr, voxel_map)
    # fmriviz.plot_slices(snr_img, "SNR_P%d" % (participant))

    predictions = all_pairs[participant-1][0]
    true_images = all_pairs[participant-1][1]
    test(snr, predictions, true_images)

#%% ---------------------------------------------------------------------
p1_gen = all_pairs[0][0][0,0]
p1_real = all_pairs[0][1][0,0]
sample_image = fmriviz.prepare_image(p1_gen, voxel_map)
fmriviz.plot_slices(sample_image)

# %%
sample_image = fmriviz.prepare_image(p1_real, voxel_map)
fmriviz.plot_slices(sample_image)

# %%
top500 = postprocess.get_top_voxels(snr,500)
binary = np.full(snr.shape, 0.2)
binary[top500] = 1
snr_binimg = fmriviz.prepare_image(binary, voxel_map)
fmriviz.plot_slices(snr_binimg, "SNR_bin_P%d" % (participant), cmap="gray_r")

# %%
top500 = postprocess.get_top_voxels(snr,1000)
binary = np.full(snr.shape, 0.2)
binary[top500] = 1
snr_binimg = fmriviz.prepare_image(binary, voxel_map)
fmriviz.plot_slices(snr_binimg, "SNR_bin1k_P%d" % (participant), cmap="gray_r")

# %%
