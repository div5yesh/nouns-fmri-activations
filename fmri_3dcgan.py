# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# %%
# example of training an conditional gan on the fashion mnist dataset
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from numpy import asarray
import numpy as np
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
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(suppress=True)


# %%
from utils.visualize import fmriviz
from utils.preprocess import dataloader, preprocess, postprocess


# %%
# define the standalone discriminator model
def define_discriminator(in_shape=(51, 61, 23, 1), n_classes=60):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
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
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	print(model.summary())
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=60):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
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
	model = Model([in_lat, in_label], out_layer)
	print(model.summary())
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
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
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=20):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
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
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
	# save the generator model
	g_model.save(os.path.join('pretrained','fmri3dcgan_1k.h5'))

# generate images
def generate_pred_pairs(model):
    latent_points, labels = generate_latent_points(1000, 2)
    X  = model.predict([latent_points, labels])
    predictions = X[:,:,:,:,0]
    return predictions, labels

# test and calculate accuracy
def test(snr, predictions, true_images):
    arr_similarity = []
    for i in range(len(predictions)):
        similarity = postprocess.evaluate(snr, predictions[i], true_images[i])
        arr_similarity += [similarity]

    accuracy = sum(arr_similarity * 1)/len(arr_similarity)
    print('Accuracy: %f' % (accuracy))
    return arr_similarity

def load_real_samples(samples, voxel_map):
	images = []
	data_flat_mean = np.mean(samples, axis=0)
	data_flat = samples - data_flat_mean
	for raw in data_flat:
		img = fmriviz.prepare_image(raw, voxel_map)
		images += [img]

	images = np.array(images)
	X = expand_dims(images, axis=-1)
	return X #(60, 51, 61, 23) (60,)


# %%
participant = 1
samples = dataloader.data[participant].samples
voxel_map = dataloader.data[participant].voxel_map
trial_map = dataloader.data[participant].trial_map
features = dataloader.features
labels = dataloader.data[participant].labels

lencoder = preprocessing.LabelEncoder()
lencoder.fit(labels)
Y = lencoder.transform(labels)


# %%
# size of the latent space
latent_dim = 1000
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
trainX = load_real_samples(samples, voxel_map)
dataset = [trainX, Y]
# train model
# train(g_model, d_model, gan_model, dataset, latent_dim, 1000)

# %%
def transform_fake_images(fake, voxel_map):
    predictions = []
    for img in fake:
        vector = postprocess.img2vector(img, voxel_map)
        predictions += [vector]
    return np.array(predictions)

# load model
model = load_model(os.path.join('pretrained','fmri3dcgan_1k.h5'))
snr = preprocess.get_snr(participant, samples, trial_map)

#%%
all_predictions = []
for i in range(100):
    fake_images, predy = generate_pred_pairs(model)
    predictions = transform_fake_images(fake_images, voxel_map)
    true_images = samples[predy]
#     print(lencoder.inverse_transform(predy))

    all_predictions += [[predictions, true_images]]

all_predictions = np.array(all_predictions)


# %%
asimtemp = test(snr, all_predictions[:,0], all_predictions[:,1])


# %%
fmriviz.plot_slices(fake_images[0], '3dconvtest1000')
thevec = postprocess.img2vector(fake_images[0],voxel_map)
theimg = fmriviz.prepare_image(thevec, voxel_map)
fmriviz.plot_slices(theimg, '3dconvtest1000_remapped')
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
