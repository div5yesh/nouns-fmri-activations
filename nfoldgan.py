#%%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os, pickle, argparse, logging
from itertools import groupby, combinations

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', default=2, type=int)
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-m', '--model', default='model')
parser.add_argument('-p', '--participant', default=1, type=int)
parser.add_argument('-g', '--gpu', default='0')
parser.add_argument('-d', '--delta', default=0.35, type=float)
parser.add_argument('-f', '--folds', default=0, type=int)
parser.add_argument('-t', '--train', dest='train', action='store_true')
# args = parser.parse_args(['-m','model_5fold15k'])
args = parser.parse_args()
print(args)

logging.basicConfig(filename=args.model+'.csv',level=logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# %%
import numpy as np
from numpy import expand_dims
from numpy import zeros, ones, ones_like
from numpy import asarray
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as kb
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
np.set_printoptions(suppress=True)

# %%
from model import SRGAN as GANModel
from nfoldtest import Test
from utils.visualize import fmriviz
from utils.preprocess import dataloader, preprocess, postprocess

# %%
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
train_vectors, embeddings = preprocess.prepare_data(features, trial_map, samples, nouns)
X = prepare_images(train_vectors, voxel_map)
snr = preprocess.get_snr(participant, samples, trial_map)
snr_img = fmriviz.prepare_image(snr, voxel_map)

#%%
# dataset = np.array([X, Y])
model_factory = GANModel(logging, embeddings, latent_dim=1000)
optimizer = Adam(lr=0.0002, beta_1=0.5)

#CGAN
# loss_weights = [1]
# losses = ['binary_crossentropy']

#CGAN HL High
# losses = ['binary_crossentropy', huber_loss(delta=args.delta)]
# loss_weights = [1e-3, 1]

# #CGAN HL Low
# losses = ['binary_crossentropy', huber_loss(delta=args.delta)]
# loss_weights = [1e-2, 1]

# #CGAN PL High
# losses = ['binary_crossentropy', perceptual_loss]
# loss_weights = [1e-3, 1]

# #CGAN PL Low
# losses = ['binary_crossentropy', perceptual_loss]
# loss_weights = [1e-2, 1]

# logging.info('***********************Hyper-Parameters: ' + str(losses) + ", " + str(loss_weights))

# %%
idx = -1
predictions = np.zeros((1, samples.shape[1]))
testobj = Test(snr, voxel_map, latent_dim=1000)

if args.folds:
	kfold = KFold(args.folds, True, 1)
	for train_idx, test_idx in kfold.split(range(60)):
		idx += 1
		# logging.info(train_idx +"; "+ test_idx +"; + idx)
		# if idx < 27:
		# 	continue
		model_name = args.model + '_fold' + str(idx) + '_p' + str(args.participant)
		if args.train:
			dataset = [X[train_idx], Y[train_idx]]
			g_model, d_model, gan_model = model_factory.create(optimizer, losses, loss_weights)
			model_factory.train(model_name, g_model, d_model, gan_model, dataset, args.epoch, args.batch)
		else:
			dataset = [train_vectors[test_idx], Y[test_idx]]
			predX = testobj.predict(model_name, dataset[1])
			predictions = np.concatenate((predictions, predX), axis=0)
		
else:
	model_name = args.model + '_p' + str(args.participant)
	if args.train:
		dataset = [X, Y]
		g_model, d_model, gan_model = model_factory.create(optimizer, losses, loss_weights)
		model_factory.train(model_name, g_model, d_model, gan_model, dataset, args.epoch, args.batch)
	else:
		dataset = [train_vectors, Y]
		predX = testobj.predict(model_name, dataset[1])
		predictions = np.concatenate((predictions, predX), axis=0)	

if not args.train:
	predictions = predictions[1:]
	testobj.test(predictions, train_vectors)

#%%
# from tensorflow.keras.utils import plot_model
# plot_model(d_model, to_file="dis.png", show_shapes=True)
# plot_model(gen_model, to_file="gen.png", show_shapes=True)

# %%
