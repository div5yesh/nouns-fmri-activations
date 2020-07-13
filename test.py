#%%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os, argparse
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='model')
parser.add_argument('-p', '--participant', default=1, type=int)
parser.add_argument('-g', '--gpu', default='0')
# args = parser.parse_args(['-m','model_less_rtlhb_p1','-r','1','-p','1'])
args = parser.parse_args()
print(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# example of training an conditional gan on the fashion mnist dataset
import numpy as np
import tensorflow as tf
from numpy.random import randint, randn
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

from utils.visualize import fmriviz
from utils.preprocess import dataloader, preprocess, postprocess

##%%
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=60):
	z_input = tf.random.normal((n_samples, latent_dim), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
	# generate labels
	ix = randint(0, n_classes, n_samples)
	return [z_input, ix]

# def prepare_images(vecs, voxel_map):
# 	images = []
# 	for raw in vecs:
# 		img = fmriviz.prepare_image(raw, voxel_map)
# 		images += [img]

# 	images = np.array(images)
# 	X = expand_dims(images, axis=-1)
# 	return X

participant = args.participant
samples = dataloader.data[participant].samples
voxel_map = dataloader.data[participant].voxel_map
trial_map = dataloader.data[participant].trial_map
features = dataloader.features
labels = dataloader.data[participant].labels

# Note: very important to have correct labels array
nouns = list(trial_map.keys())
lencoder = preprocessing.LabelEncoder()
Y = lencoder.fit_transform(nouns)

train_vectors, embeddings = preprocess.prepare_data(features,trial_map,samples,nouns)
snr = preprocess.get_snr(participant, samples, trial_map)
# snr_img = fmriviz.prepare_image(snr, voxel_map)

# size of the latent space
latent_dim = 1000
def transform_fake_images(fake, voxel_map):
    predictions = []
    for img in fake:
        vector = postprocess.img2vector(img, voxel_map)
        predictions += [vector]
    return np.array(predictions)

def test(snr, combinations, predictions, true_images):
    arr_similarity = []
    for pair in combinations:
        idx = list(pair)
        similarity = postprocess.evaluate(snr, predictions[idx], true_images[idx])
        arr_similarity += [similarity]

    return np.array(arr_similarity)

def cl_eval(snr, predictions, true_images, top=500):
    arr_similarity = []
    for i in range(len(predictions)):
        similarity = postprocess.classic_eval(snr, predictions[i], true_images[i], 0.7, top)
        # similarity = postprocess.classic_eval(true_images[i], predictions[i], true_images[i], 0.7, 500)
        arr_similarity += [similarity]

    accuracy = sum(arr_similarity * 1)/len(arr_similarity)
    print('Accuracy: %f' % (accuracy))
    return np.array(arr_similarity)

#%%
# load model
model = load_model(os.path.join('pretrained', args.model + '.h5'))

#%%----------------------------------------test------------------------------------------
test_combinations = list(combinations(Y, 2))
predictions = np.zeros((1,samples.shape[1]))
latent_points, _ = generate_latent_points(1000, len(Y))

for i in range(6):
    start = i * 10
    end = (i + 1) * 10
    X  = model.predict([latent_points[start:end], Y[start:end]])
    fake_image = X[:,:,:,:,0]
    preds = transform_fake_images(fake_image, voxel_map)
    predictions = np.concatenate((predictions, preds), axis=0)

predictions = predictions[1:]
true_vecs = train_vectors

arr_similarity = test(snr, test_combinations, predictions, true_vecs)
accuracy = np.mean(arr_similarity)
print('Accuracy: %f' % (accuracy))

temp = cl_eval(snr, predictions, true_vecs)

# %% --------------------------------main plots-------------------------------------------------------------------
sample_idx = 15
# vmin = np.min(predictions[sample_idx])/2
# vmax = np.max(predictions[sample_idx])/2
vmin = np.min(true_vecs[sample_idx])
vmax = np.max(true_vecs[sample_idx])

# rescaled = predictions[sample_idx] * 5

theimg = fmriviz.prepare_image(predictions[sample_idx], voxel_map, fill=vmin)
fmriviz.plot_slices(theimg, vmin, vmax)

# theimg = fmriviz.prepare_image(rescaled, voxel_map, fill=vmin)
# fmriviz.plot_slices(theimg, vmin, vmax)

# tvox = postprocess.get_top_voxels(predictions[0], 500)
# binary = np.full(true_vecs[0].shape, -0.2)
# binary[tvox] = 1

# theimg = fmriviz.prepare_image(binary, voxel_map, fill=-1)
# fmriviz.plot_slices(theimg, -1, 1, cmap='gray_r')

# tvox = postprocess.get_top_voxels(snr, 500)
# binary = np.full(true_vecs[0].shape, -0.2)
# binary[tvox] = 1

# theimg = fmriviz.prepare_image(binary, voxel_map, fill=-1)
# fmriviz.plot_slices(theimg, -1, 1, cmap='gray_r')

# tvox = postprocess.get_top_voxels(true_vecs[0], 500)
# binary = np.full(true_vecs[0].shape, -0.2)
# binary[tvox] = 1

# theimg = fmriviz.prepare_image(binary, voxel_map, fill=-1)
# fmriviz.plot_slices(theimg, -1, 1, cmap='gray_r')

vmin = np.min(true_vecs[sample_idx])
vmax = np.max(true_vecs[sample_idx])

theimg = fmriviz.prepare_image(true_vecs[sample_idx], voxel_map, fill=vmin)
fmriviz.plot_slices(theimg, vmin, vmax)

#%%
# all_predictions = []
# custom_labels = [["cup","cup"],["hammer","hammer"],["house","house"],["knife","knife"],["screwdriver","screwdriver"]]

# all_img_pairs = []
# nouns = set(trial_map.keys())
# test_combinations = list(combinations(nouns, 2))

# for pair in test_combinations:
#     custom_y = lencoder.transform(pair)
#     # print(custom_y)
#     fake_images, predy = generate_pred_pairs(model, custom_y)
#     predictions = transform_fake_images(fake_images, voxel_map)
#     true_vecs = train_vectors[predy]
#     # print(lencoder.inverse_transform(predy))

#     all_predictions += [[predictions, true_vecs]]
#     # all_img_pairs += [[fake_images,trainX[predy,:,:,:,0], custom_labels[i]]]
# all_predictions = np.array(all_predictions)


# # %%
# asimtemp = test(snr, all_predictions[:,0], all_predictions[:,1])


# #%% ---------------------------------------------------------------------
# for i in range(5):
#     p1_gen = all_predictions[i][0][1]
#     p1_real = all_predictions[i][1][1]
#     lbl = test_combinations[i][1]

#     vmin = np.min(p1_real)
#     vmax = np.max(p1_real)

#     sample_image = fmriviz.prepare_image(p1_real, voxel_map, fill=vmin)
#     fmriviz.plot_slices(sample_image,vmin,vmax, filename="GAN_" + lbl + "_real")

#     sample_image = fmriviz.prepare_image(p1_gen, voxel_map, fill=vmin)
#     fmriviz.plot_slices(sample_image,vmin,vmax, filename="GAN_" + lbl + "_gen")

# # %%
# cosine_similarity(fake_images[0].reshape(1,-1), theimg.reshape(1,-1))


# # %%
# cosine_similarity(trainX[predy[0],:,:,:,0].reshape(1,-1), theimg.reshape(1,-1))


# # %%
# cosine_similarity(thevec.reshape(1,-1), samples[predy[0]].reshape(1,-1))


# # %%
# top = postprocess.get_top_voxels(samples[predy[0]],500)
# cosine_similarity(thevec[top].reshape(1,-1), samples[predy[0]][top].reshape(1,-1))


# # %%
# top = postprocess.get_top_voxels(snr,500)
# cosine_similarity((thevec * snr)[top].reshape(1,-1), samples[predy[0]][top].reshape(1,-1))

#%% -----------------------------------volume plots---------------------------------------------------------------------
# nouns = list(trial_map.keys())
# y_Bar = lencoder.transform(nouns)

# predictions = np.zeros((1,51,61,23))
# latent_points, _ = generate_latent_points(1000, len(y_Bar))

# for i in range(6):
# 	start = i * 10
# 	end = (i + 1) * 10
# 	X  = model.predict([latent_points[start:end], y_Bar[start:end]])
# 	fake_images = X[:,:,:,:,0]
# 	predictions = np.concatenate((predictions, fake_images), axis=0)

# predictions = predictions[1:]

#%%
# vol_match = []
# for pred in predictions:
# 	vec = postprocess.img2vector(pred, voxel_map)
# 	img = fmriviz.prepare_image(vec, voxel_map)
# 	vol_match += [cosine_similarity(img.reshape(1,-1), pred.reshape(1,-1))[0][0]]

# match = sum(vol_match * 1)/len(vol_match)
# print('Volume Match: %f' % (match))

# # %%
# vmin = np.min(true_vecs[0])
# vmax = np.max(true_vecs[0])

# vec = postprocess.img2vector(predictions[0], voxel_map)
# img = fmriviz.prepare_image(vec, voxel_map, vmin)
# fmriviz.plot_slices(img, vmin, vmax, "refrigrator_gen_3dgan_remapped")

# %%
