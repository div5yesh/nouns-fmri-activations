# %%
import os, argparse
from itertools import combinations

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--model', default='model')
# parser.add_argument('-p', '--participant', default=1, type=int)
# parser.add_argument('-g', '--gpu', default='0')
# args = parser.parse_args(['-m','model_less_rtlhb_p1','-r','1','-p','1'])
# print(args)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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

#%%
class Test:
    def __init__(self, snr, voxel_map, latent_dim):
        self.snr = snr
        self.voxel_map = voxel_map
        self.latent_dim = latent_dim
        self.n_voxels = voxel_map[0].shape[0]

    def generate_latent_points(self, latent_dim, n_samples):
        z_input = tf.random.normal((n_samples, latent_dim), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
        return z_input

    def transform_fake_images(self, fake):
        predictions = []
        for img in fake:
            vector = postprocess.img2vector(img, self.voxel_map)
            predictions += [vector]
        return np.array(predictions)

    def evaluate(self, snr, combinations, predictions, dataset):
        true_images, Y = dataset
        arr_similarity = []
        for pair in combinations:
            idx = list(pair)
            # print(Y[idx])
            similarity = postprocess.evaluate(snr, predictions[idx], true_images[idx])
            arr_similarity += [similarity]

        accuracy = np.mean(arr_similarity)
        print('Match Metric: %f' % (accuracy))
        return np.array(arr_similarity)

    def classic_eval(self, snr, predictions, dataset, top=500):
        true_images, Y = dataset
        arr_similarity = []
        for i in range(len(predictions)):
            similarity = postprocess.classic_eval(snr, predictions[i], true_images[i], 0.7, top)
            arr_similarity += [similarity]

        accuracy = sum(arr_similarity * 1)/len(arr_similarity)
        print('Cosine Metric: %f' % (accuracy))
        return np.array(arr_similarity)

    def predict_test(self, model_name, dataset):
        Y = dataset[1]
        n_classes = len(Y)
        n_batch = int(n_classes / 2)
        predictions = np.zeros((1, self.n_voxels))

        test_combinations = list(combinations(range(n_classes), 2))
        latent_points = self.generate_latent_points(self.latent_dim, n_classes)
        model = load_model(os.path.join('pretrained', model_name + '.h5'))

        for i in range(n_batch):
            start = i * 2
            end = (i + 1) * 2
            X  = model.predict([latent_points[start:end], Y[start:end]])
            fake_image = X[:,:,:,:,0]
            preds = self.transform_fake_images(fake_image)
            predictions = np.concatenate((predictions, preds), axis=0)

        predictions = predictions[1:]

        match_similarity = self.evaluate(self.snr, test_combinations, predictions, dataset)
        cosine_similarity = self.classic_eval(self.snr, predictions, dataset)
        return match_similarity, cosine_similarity

#%%
# participant = args.participant
# samples = dataloader.data[participant].samples
# voxel_map = dataloader.data[participant].voxel_map
# trial_map = dataloader.data[participant].trial_map
# features = dataloader.features
# labels = dataloader.data[participant].labels

# # Note: very important to have correct labels array
# nouns = list(trial_map.keys())
# lencoder = preprocessing.LabelEncoder()
# Y = lencoder.fit_transform(nouns)
# latent_dim = 1000

# true_vectors, embeddings = preprocess.prepare_data(features,trial_map,samples,nouns)
# snr = preprocess.get_snr(participant, samples, trial_map)

#%%
# testobj = Test(snr, voxel_map, latent_dim)
# testobj.predict_test(args.model, true_vectors, Y)

#%%