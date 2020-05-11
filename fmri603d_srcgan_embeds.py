#%%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pickle
from itertools import groupby, combinations

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"


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
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer, name='discriminator')
	# compile model
	# opt = Adam(lr=0.0002, beta_1=0.5)
	# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	# print(model.summary())
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

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# get noise and label inputs from generator model
	latent_points, labels = g_model.input
	# get image output from the generator model
	generator_output = g_model.output
	# fake_features = vgg(generator_output)
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect image output and label input from generator as inputs to discriminator
	discriminator_output = d_model([generator_output, labels])
	# define gan model as taking noise and label and outputting a classification
	# model = Model([latent_points, labels], discriminator_output)
	model = Model([latent_points, labels], [discriminator_output, generator_output])
	# compile model
	# opt = Adam(lr=0.0002, beta_1=0.5)
	# model.compile(loss='binary_crossentropy', optimizer=opt)
	print(model.summary())
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
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=60):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=2):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = n_batch #int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# generate 'fake' examples
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			# prepare points in latent space as input for the generator
			[z_input, _] = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_real], [y_gan, X_real])
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f,%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss[0], g_loss[1]))
	# save the generator model
	g_model.save(os.path.join('pretrained','fmri3d_srcgan60_1k.h5'))

# generate images
def generate_pred_pairs(model, labels):
    latent_points, _ = generate_latent_points(1000, 2)
    X  = model.predict([latent_points, labels])
    predictions = X[:,:,:,:,0]
    return predictions, labels

# test and calculate accuracy
def test(snr, predictions, true_images):
    arr_similarity = []
    for i in range(len(predictions)):
        similarity = postprocess.evaluate(snr, predictions[i], true_images[i], 500)
        arr_similarity += [similarity]

    accuracy = sum(arr_similarity * 1)/len(arr_similarity)
    print('Accuracy: %f' % (accuracy))
    return arr_similarity

def prepare_images(vecs, voxel_map):
	images = []
	for raw in vecs:
		img = fmriviz.prepare_image(raw, voxel_map)
		images += [img]

	images = np.array(images)
	X = expand_dims(images, axis=-1)
	return X

def perceptual_loss(real, fake):
	snr_img = tf.reshape(snr_img, (-1,1))
	diff = tf.reshape(real, (-1,1)) - tf.reshape(fake, (-1,1))
	return kb.mean(kb.square(tf.math.multiply(diff, snr_img))


def adverserial_loss(real_output, fake_output):
	alpha = 1e-3
	g_loss = kb.mean(kb.l2_normalize(fake_output - kb.ones_like(fake_output)))
	d_loss_real = kb.mean(kb.l2_normalize(real_output - kb.ones_like(real_output)))
	d_loss_fake = kb.mean(kb.l2_normalize(fake_output + kb.zeros_like(fake_output)))
	d_loss = d_loss_fake + d_loss_real
	return (g_loss*alpha, d_loss*alpha)


# %%
participant = 1
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

# %%
optimizer = Adam(lr=0.0002, beta_1=0.5)
# create the discriminator
d_model = define_discriminator(embeddings)
d_model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
# create the generator
g_model = define_generator(embeddings, latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
gan_model.compile(loss=['binary_crossentropy', perceptual_loss], loss_weights=[1e-3, 1], optimizer=optimizer)

#%%
# train model
# train(g_model, d_model, gan_model, dataset, latent_dim, 5)

# %%
def transform_fake_images(fake, voxel_map):
    predictions = []
    for img in fake:
        vector = postprocess.img2vector(img, voxel_map)
        predictions += [vector]
    return np.array(predictions)

# load model
model = load_model(os.path.join('pretrained','fmri3d_srcgan60_1k.h5'))

#%%
all_predictions = []
custom_labels = [["cup","cup"],["hammer","hammer"],["house","house"],["knife","knife"],["screwdriver","screwdriver"]]

all_img_pairs = []
nouns = set(trial_map.keys())
test_combinations = list(combinations(nouns, 2))

for pair in test_combinations:
    custom_y = lencoder.transform(pair)
    # print(custom_y)
    fake_images, predy = generate_pred_pairs(model, custom_y)
    predictions = transform_fake_images(fake_images, voxel_map)
    true_vecs = train_vectors[predy]
    # print(lencoder.inverse_transform(predy))

    all_predictions += [[predictions, true_vecs]]
    # all_img_pairs += [[fake_images,trainX[predy,:,:,:,0], custom_labels[i]]]
all_predictions = np.array(all_predictions)


# %%
asimtemp = test(snr, all_predictions[:,0], all_predictions[:,1])


#%% ---------------------------------------------------------------------
for i in range(5):
    p1_gen = all_predictions[i][0][1]
    p1_real = all_predictions[i][1][1]
    lbl = test_combinations[i][1]

    vmin = np.min(p1_real)
    vmax = np.max(p1_real)

    sample_image = fmriviz.prepare_image(p1_real, voxel_map, fill=vmin)
    fmriviz.plot_slices(sample_image,vmin,vmax, filename="GAN_" + lbl + "_real")

    sample_image = fmriviz.prepare_image(p1_gen, voxel_map, fill=vmin)
    fmriviz.plot_slices(sample_image,vmin,vmax, filename="GAN_" + lbl + "_gen")


# %%
fmriviz.plot_slices(fake_images[0], '3dconvtest1k_embeds')
thevec = postprocess.img2vector(fake_images[0],voxel_map)
theimg = fmriviz.prepare_image(thevec, voxel_map)
fmriviz.plot_slices(theimg, '3dconvtest1k_remap_embeds')
fmriviz.plot_slices(trainX[predy[0],:,:,:,0])


# %%
cosine_similarity(fake_images[0].reshape(1,-1), theimg.reshape(1,-1))


# %%
cosine_similarity(trainX[predy[0],:,:,:,0].reshape(1,-1), theimg.reshape(1,-1))


# %%
cosine_similarity(thevec.reshape(1,-1), samples[predy[0]].reshape(1,-1))


# %%
top = postprocess.get_top_voxels(samples[predy[0]],500)
cosine_similarity(thevec[top].reshape(1,-1), samples[predy[0]][top].reshape(1,-1))


# %%
top = postprocess.get_top_voxels(snr,500)
cosine_similarity((thevec * snr)[top].reshape(1,-1), samples[predy[0]][top].reshape(1,-1))


# %%
#%%----------------------------------------test------------------------------------------
predictions = np.zeros((1,21764))
test_combinations = list(combinations(Y, 2))
latent_points, _ = generate_latent_points(1000, len(Y))

for i in range(6):
	start = i * 10
	end = (i + 1) * 10
	X  = model.predict([latent_points[start:end], Y[start:end]])
	fake_image = X[:,:,:,:,0]
	preds = transform_fake_images(fake_image, voxel_map)
	predictions = np.concatenate((predictions, preds), axis=0)

predictions = predictions[1:]
true_vecs = train_vectors[Y]

#%%
def test(snr, combinations, predictions, true_images):
    arr_similarity = []
    for pair in combinations:
        idx = list(pair)
        similarity = postprocess.evaluate(snr, predictions[idx], true_images[idx])
        arr_similarity += [similarity]

    accuracy = sum(arr_similarity * 1)/len(arr_similarity)
    print('Accuracy: %f' % (accuracy))
    return np.array(arr_similarity)

temp = test(snr, test_combinations, predictions, true_vecs)

#%%
def cl_eval(snr, predictions, true_images):
    arr_similarity = []
    for i in range(len(predictions)):
        # similarity = postprocess.evaluate(snr, predictions[i], true_images[i])
        similarity = postprocess.classic_eval(snr, predictions[i], true_images[i], 0.7, 500)
        # similarity = postprocess.classic_eval(true_images[i], predictions[i], true_images[i], 0.7, 500)
        arr_similarity += [similarity]

    accuracy = sum(arr_similarity * 1)/len(arr_similarity)
    print('Accuracy: %f' % (accuracy))
    return np.array(arr_similarity)

temp = cl_eval(snr, predictions, true_vecs)

# %%

vmin = np.min(true_vecs[0])
vmax = np.max(true_vecs[0])

theimg = fmriviz.prepare_image(predictions[0], voxel_map, fill=vmin)
fmriviz.plot_slices(theimg, vmin, vmax)

# tvox = postprocess.get_top_voxels(predictions[0], 500)
# binary = np.full(true_vecs[0].shape, -0.2)
# binary[tvox] = 1

# theimg = fmriviz.prepare_image(binary, voxel_map, fill=-1)
# fmriviz.plot_slices(theimg, -1, 1, cmap='gray_r')

# tvox = postprocess.get_top_voxels(true_vecs[0], 500)
# binary = np.full(true_vecs[0].shape, -0.2)
# binary[tvox] = 1

# theimg = fmriviz.prepare_image(binary, voxel_map, fill=-1)
# fmriviz.plot_slices(theimg, -1, 1, cmap='gray_r')

theimg = fmriviz.prepare_image(true_vecs[0], voxel_map, fill=vmin)
fmriviz.plot_slices(theimg, vmin, vmax)

#%%
nouns = list(trial_map.keys())
y_Bar = lencoder.transform(nouns)

predictions = np.zeros((1,51,61,23))
latent_points, _ = generate_latent_points(1000, len(y_Bar))

for i in range(6):
	start = i * 10
	end = (i + 1) * 10
	X  = model.predict([latent_points[start:end], y_Bar[start:end]])
	fake_images = X[:,:,:,:,0]
	predictions = np.concatenate((predictions, fake_images), axis=0)

predictions = predictions[1:]

#%%
vol_match = []
for pred in predictions:
	vec = postprocess.img2vector(pred, voxel_map)
	img = fmriviz.prepare_image(vec, voxel_map)
	vol_match += [cosine_similarity(img.reshape(1,-1), pred.reshape(1,-1))[0][0]]

match = sum(vol_match * 1)/len(vol_match)
print('Volume Match: %f' % (match))

# %%
vmin = np.min(true_vecs[0])
vmax = np.max(true_vecs[0])

vec = postprocess.img2vector(predictions[0], voxel_map)
img = fmriviz.prepare_image(vec, voxel_map, vmin)
fmriviz.plot_slices(img, vmin, vmax, "refrigrator_gen_3dgan_remapped")

genimg = predictions[1]

for i in range(genimg.shape[0]):
	for j in range(genimg.shape[1]):
		for k in range(genimg.shape[2]):
			if -0.05 < genimg[i][j][k] < 0.05:
				genimg[i][j][k] = vmin

# theimg = fmriviz.prepare_image(predictions[0], voxel_map, fill=vmin)
fmriviz.plot_slices(genimg, vmin, vmax)

# tvox = postprocess.get_top_voxels(predictions[0], 500)
# binary = np.full(true_vecs[0].shape, -0.2)
# binary[tvox] = 1

# theimg = fmriviz.prepare_image(binary, voxel_map, fill=-1)
# fmriviz.plot_slices(theimg, -1, 1, cmap='gray_r')

# tvox = postprocess.get_top_voxels(true_vecs[0], 500)
# binary = np.full(true_vecs[0].shape, -0.2)
# binary[tvox] = 1

# theimg = fmriviz.prepare_image(binary, voxel_map, fill=-1)
# fmriviz.plot_slices(theimg, -1, 1, cmap='gray_r')

theimg = fmriviz.prepare_image(true_vecs[0], voxel_map, fill=vmin)
fmriviz.plot_slices(theimg, vmin, vmax)

# %%
# _, x_phi = self.vgg.build_model(x, tf.constant(False), False)
	# _, imitation_phi = self.vgg.build_model(
	# 	imitation, tf.constant(False), True)
	# content_loss = None
	# for i in range(len(x_phi)):
	# 	l2_loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
	# 	if content_loss is None:
	# 		content_loss = l2_loss
	# 	else:
	# 		content_loss += l2_loss
	# return tf.reduce_mean(content_loss)