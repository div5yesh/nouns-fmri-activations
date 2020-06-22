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
from utils.visualize import fmriviz
from utils.preprocess import dataloader, preprocess, postprocess

# %%
# define the standalone discriminator model
def define_discriminator(embeddings, in_shape=(51, 61, 23, 1), n_classes=60):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	embedding_layer = Embedding(input_dim=n_classes, output_dim=25, input_length=1, weights=[embeddings], trainable=False)
	li = embedding_layer(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1] * in_shape[2]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], in_shape[2], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv3D(32, (3,3,3), strides=(2,2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv3D(64, (3,3,3), strides=(2,2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv3D(128, (3,3,3), strides=(2,2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(2048, activation=None)(fe)
	# define model
	model = Model([in_image, in_label], out_layer, name='discriminator')
	# compile model
	# opt = Adam(lr=0.0002, beta_1=0.5)
	# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(embeddings, latent_dim, n_classes=60):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	embedding_layer = Embedding(input_dim=n_classes, output_dim=25, input_length=1, weights=[embeddings], trainable=False)
	li = embedding_layer(in_label)
	# linear multiplication
	n_nodes = 7 * 7 * 7
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 7, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x21
	gen = Conv3DTranspose(128, (3,3,3), strides=(2,3,1), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# # upsample to 28x21
	gen = Conv3DTranspose(128, (3,3,3), strides=(2,1,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# # upsample to 56x63
	gen = Conv3DTranspose(128, (3,3,3), strides=(2,3,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output - 51x51x23
	out_layer = Conv3D(1, (6,3,6), strides=(1,1,1), activation='tanh', padding='valid')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer, name='generator')
	# print(model.summary())
	return model

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
	# y = randint(7, 12, (n_samples, 1)) / 10
	return X, labels#], y

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
def generate_fake_samples(generator, latent_dim, n_samples, labels_input):
	# generate points in latent space
	# z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	z_input, z_labels = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	# y = zeros((n_samples, 1))
	# y = randint(0, 3, (n_samples, 1)) / 10
	return images #, labels_input], y

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

def difference(inputs):
	x1, x2 = inputs
	return (x1 - x2)

#%%
input_shape = (51, 61, 23, 1)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
in_label_a = Input(shape=(1,))
in_label_b = Input(shape=(1,))

d_model = define_discriminator(embeddings)
g_model = define_generator(embeddings, latent_dim)

# construct dis model
dis_d1 = d_model([input_a, in_label_a])
dis_d2 = d_model([input_b, in_label_b])

dis_dif_layer = Lambda(difference)([dis_d1, dis_d2])
dis_out_layer = Dense(1, activation='sigmoid')(dis_dif_layer)

dis_model = Model([input_a, input_b, in_label_a, in_label_b], outputs=dis_out_layer, name="dis_model")
print(dis_model.summary())
dis_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# construct gen model
input_c = Input(shape=input_shape)
in_label_c = Input(shape=(1,))
gen_z_input, gen_label_input = g_model.input
d_model.trainable = False
gen_d1 = d_model([input_c, in_label_c])
# gen_d1.trainable = False
gen_d2 = d_model([g_model.output, gen_label_input])
# gen_d2.trainable = False

gen_dif_layer = Lambda(difference)([gen_d1, gen_d2])
gen_out_layer = Dense(1, activation='sigmoid')(gen_dif_layer)

gen_model = Model([input_c, gen_z_input, in_label_c, gen_label_input], outputs=[gen_out_layer, g_model.output], name="gan_model")
print(gen_model.summary())
gen_model.compile(loss=['binary_crossentropy', perceptual_loss], optimizer=optimizer, metrics=['accuracy'])

# %%
def train(g_model, dis_model, gen_model, dataset, latent_dim, n_epochs=100, n_batch=2):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			y_real = randint(7, 12, (n_batch, 1)) / 10
			X_real, labels_real = generate_real_samples(dataset, n_batch)
			X_fake = generate_fake_samples(g_model, latent_dim, n_batch, labels_real)

			d_loss, _ = dis_model.train_on_batch([X_real, X_fake, labels_real, labels_real], y_real)

			z_input, z_labels = generate_latent_points(latent_dim, n_batch)
			y_fake = randint(0, 3, (n_batch, 1)) / 10

			g_loss, _ = gen_model.train_on_batch([dataset[0][z_labels], z_input, z_labels, z_labels], [y_fake, dataset[0][labels]])
			
			print('>%d, %d/%d, d=%.3f, g=%.3f, %.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss[0], g_loss[1]))
	# save the generator model
	g_model.save(os.path.join('pretrained', args.model + '_p' + str(args.participant) + '.h5'))

# %%
train(g_model, dis_model, gen_model, dataset, latent_dim, args.epoch, args.batch)

#%%
# from tensorflow.keras.utils import plot_model
# plot_model(dis_model, to_file="dis.png", show_shapes=True)
# plot_model(gen_model, to_file="gen.png", show_shapes=True)