# %%
from itertools import groupby, combinations
import numpy as np
import scipy.io, pickle, os
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression

from utils.visualize import fmriviz
from utils.preprocess import dataset, preprocess, postprocess

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

    # model_path = os.path.join(os.getcwd(), 'pretrained', 'rmodel%d.h5'%(idx))
    # if not os.path.isfile(model_path):
    #     pickle.dump(reg_model, open("rmodel"+str(idx)+".h5", "wb"))

    return np.array(predictions), np.array(true_images)

# %%
participant = 1
samples = dataset.data[participant].samples
voxel_map = dataset.data[participant].voxel_map
trial_map = dataset.data[participant].trial_map
features = dataset.features

snr = preprocess.get_snr(participant, samples, trial_map)
snr_img = fmriviz.prepare_image(snr, voxel_map)
fmriviz.plot_slices(snr_img, "SNR_P%d" % (participant))

# %%
predictions, true_images = train(features, trial_map, samples)

#%%
similarity = postprocess.evaluate(snr, predictions, true_images)
accuracy = sum(similarity * 1)/len(similarity)
print('Accuracy: %f' % (accuracy))

#%% ---------------------------------------------------------------------
sample_image = fmriviz.prepare_image(predictions[0][0], voxel_map)
fmriviz.plot_slices(sample_image)

# %%
sample_image = fmriviz.prepare_image(true_images[0][0], voxel_map)
fmriviz.plot_slices(sample_image)

# %%