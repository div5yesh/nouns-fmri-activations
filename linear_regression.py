# %%
from itertools import groupby, combinations
import numpy as np
import scipy.io, pickle, os
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from utils.visualize import fmriviz
from utils.preprocess import dataset, preprocess

# %%
def find_top500voxels(voxel_map):
    stability_scores = []
    for i in range(len(voxel_map)):
        correlation_mat = np.corrcoef(voxel_map[i].T)
        correlation = (np.sum(correlation_mat) - 58)/2
        avg_correlation = correlation/1653
        stability_scores += [avg_correlation]

    stability_scores = np.array(stability_scores)
    top500 = stability_scores.argsort()[-500:][::-1]
    return top500

def select(voxels, image):
    img = []
    for i in range(len(image)):
        img += [image[i,:][voxels]]

    return np.array(img)

def match(pred, act):
    similarity = cosine_similarity(pred, act)
    #p1i1_p2i2
    self_match = np.sum(np.diag(similarity))    
    #p1i2_p2i1
    cross_match = np.sum(similarity) - self_match
    return self_match > cross_match

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

    return predictions, true_images

def evaluate(snr, predictions, true_images):
    similarity_map = []

    for i in range(len(predictions)):

        similarity = match(predictions[i] * snr, true_images[i])
        similarity_map += [similarity]
        
        # print('Eval Combination: %d' % (i))

    return np.array(similarity_map)

# %%
samples = dataset.data[1].samples
voxel_map = dataset.data[1].voxel_map
trial_map = dataset.data[1].trial_map
features = dataset.features
snr = preprocess.get_snr(1, samples, trial_map)
sample_img = fmriviz.prepare_image(samples[0], voxel_map)
snr_img = fmriviz.prepare_image(snr, voxel_map)
fmriviz.plot_slices(sample_img)

# %%
predictions, true_images = train(features, trial_map, samples)

#%%
similarity = evaluate(snr, predictions, true_images)
accuracy = sum(similarity * 1)/len(similarity)
print('Accuracy: %f' % (accuracy))

#%% ---------------------------------------------------------------------
sample_image = fmriviz.prepare_image(predictions[0][0], voxel_map)
fmriviz.plot_slices(sample_image)

# %%
sample_image = fmriviz.prepare_image(true_images[0][0], voxel_map)
fmriviz.plot_slices(sample_image)

# %%