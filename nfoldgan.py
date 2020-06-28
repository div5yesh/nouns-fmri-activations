#%%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os, pickle, argparse
from itertools import groupby, combinations

#%%
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', default=2, type=int)
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-m', '--model', default='model')
parser.add_argument('-p', '--participant', default=1, type=int)
parser.add_argument('-g', '--gpu', default='0')
parser.add_argument('-d', '--delta', default=0.24, type=float)
args = parser.parse_args()
print(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# %%
# example of training an conditional gan on the fashion mnist dataset
from numpy import expand_dims
from numpy import zeros
from numpy import ones, ones_like
from numpy.random import randn
from numpy.random import randint
from numpy import asarray
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as kb
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)

# %%
from model import GAN
from utils.visualize import fmriviz
from utils.preprocess import dataloader, preprocess, postprocess

# %%
# select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	# y = ones((n_samples, 1))
	y = randint(7, 12, (n_samples, 1)) / 10
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=60):
	# generate points in the latent space
	# x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	# z_input = x_input.reshape(n_samples, latent_dim)
	z_input = tf.random.normal((n_samples, latent_dim), mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
	# generate labels
	z_labels = randint(0, n_classes, n_samples)
	return z_input, z_labels
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, z_labels = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, z_labels])
	# create class labels
	# y = zeros((n_samples, 1))
	y = randint(0, 3, (n_samples, 1)) / 10
	return [images , z_labels], y

def prepare_images(vecs, voxel_map):
	images = []
	for raw in vecs:
		img = fmriviz.prepare_image(raw, voxel_map)
		images += [img]

	images = np.array(images)
	X = expand_dims(images, axis=-1)
	return X

def perceptual_loss(real, fake):
	b_size = tf.shape(real)[0]
	ranks = tf.cast(tf.reshape(snr_img, (1,-1)),tf.float32)
	diff = tf.reshape(real, (b_size, -1)) - tf.reshape(fake, (b_size, -1))
	weighted = tf.math.multiply(diff, ranks)
	return kb.mean(kb.square(weighted))

def huber_loss(delta):
	def huber_fn(real, fake):
		b_size = tf.shape(real)[0]
		ranks = tf.cast(tf.reshape(snr_img, (1,-1)),tf.float32)
		diff = tf.reshape(real, (b_size, -1)) - tf.reshape(fake, (b_size, -1))
		weighted = tf.math.multiply(diff, ranks)
		return kb.mean(kb.sqrt(kb.square(weighted) + delta * delta))
	return huber_fn

# %%
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

#%%
train_vectors, embeddings = preprocess.prepare_data(features,trial_map,samples,nouns)
trainX = prepare_images(train_vectors,voxel_map)
snr = preprocess.get_snr(participant, samples, trial_map)
snr_img = fmriviz.prepare_image(snr, voxel_map)

# size of the latent space
latent_dim = 1000
dataset = [trainX, Y]
optimizer = Adam(lr=0.0002, beta_1=0.5)
model = GAN(embeddings, latent_dim)

# %%
def train(g_model, dis_model, gen_model, dataset, latent_dim, n_epochs=100, n_batch=2):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# y_real = randint(7, 12, (n_batch, 1)) / 10
			[X_real, labels_real], y_real = generate_real_samples(dataset, n_batch)
			[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, n_batch)

			d_loss, _ = dis_model.train_on_batch([X_real, X_fake, labels_real, labels_fake], y_real)

			z_input, z_labels = generate_latent_points(latent_dim, n_batch)
			# y_fake = randint(0, 3, (n_batch, 1)) / 10

			g_loss = gen_model.train_on_batch([dataset[0][z_labels], z_input, z_labels, z_labels], [y_fake, dataset[0][z_labels]])
			
			print('>%d, %d/%d, d=%.3f, g=%.3f, %.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss[0], g_loss[1]))
	# save the generator model
	g_model.save(os.path.join('pretrained', args.model + '_p' + str(args.participant) + '.h5'))

# %%
train(g_model, dis_model, gen_model, dataset, latent_dim, args.epoch, args.batch)

#%%
# from tensorflow.keras.utils import plot_model
# plot_model(dis_model, to_file="dis.png", show_shapes=True)
# plot_model(gen_model, to_file="gen.png", show_shapes=True)