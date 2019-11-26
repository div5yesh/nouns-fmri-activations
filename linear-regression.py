# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
from itertools import groupby
import numpy as np
import scipy.io
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

# %%
mat = scipy.io.loadmat('data-science-P1.mat')
meta = mat["meta"]
info = mat["info"]
data = mat["data"]

# %%
voxel_map = meta["coordToCol"][0][0]    #(51, 61, 23)
voxel_vec = meta["colToCoord"][0][0]    #(21764, 3)
scan_shape = voxel_map.shape

# %%
fmri_image = np.zeros(scan_shape)
# for n in range(10):
for i in range(scan_shape[0]):
    for j in range(scan_shape[1]):
        for k in range(scan_shape[2]):
            voxel = voxel_map[i][j][k]
            fmri_image[i][j][k] = data[0,0][0][voxel - 1]

# %%
for i in range(scan_shape[2]):
    plt.imshow(fmri_image[:,:,i])
    plt.show()

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxel_map, facecolor=[0,0,0])
plt.show()

# %%
fmri_images = np.zeros((1,21764))
features = np.zeros((1,25))
weights = fmri_images.T @ features 

# %%
weights = np.zeros((21764,25))
features = np.zeros((1,25))
fmri_images = features @ weights.T

# %% [markdown]
# Snippet for voxel image plot

# %%
# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxels = cube1 | cube2 | link

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k')

plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# your real data here - some 3d boolean array
x, y, z = np.indices((10, 10, 10))
voxels = (x == y) | (y == z)

ax.voxels(voxels, facecolors=["green"], edgecolor="k")

plt.show() 

# %%
fp = open("features.txt", "r")
feat_data = fp.read()
fp.close()

# %%
features = dict()
words = feat_data.split("\n\n\n")
for word in words:
    feature = word.split("\n\n")
    value = list(map(lambda x:float(x.split(" ")[1]), sorted(feature[1].split("\n"), key=lambda x: x[0])))
    features[feature[0]] = value

# TODO: index map to same feature
# features = np.asarray(feature_values)   #(60, 25)

# %%
trial_map = dict()
for i in range(len(info[0])):
    trial = info[0][i]
    condition_idx = trial['cond_number'][0][0]
    word_idx = trial['word_number'][0][0]
    word = trial['word'][0]
    key = (condition_idx, word_idx, word)
    trial_map[word] = trial_map.get(word, []) + [i]

# %%
samples = 360 #len(data)
images = np.zeros((samples,21764))    #(360, 21764)
for i in range(samples):
    images[i] = data.flatten()[i].flatten()

mean_image = np.mean(images, axis=0)
images = images - mean_image

#%%
X = []
Y = []
for key in trial_map:
    img = trial_map[key]
    for i in img:
        X += [features[key]]
        Y += [images[i]]

#%%
# TODO: 
# remove mean from images - 
# predict word from feature vector - 
# add data points for each of the 360 images - 
# train on each participant separately - 
# train on combined data set - 
# visualize predicted images - 
# use cosine similarity - done

# %%
X = np.asarray(X)
Y = np.asarray(Y)
N = len(images)
split_idx = int(N * .75)

trainX = X[:split_idx]
trainY = Y[:split_idx]

testX = X[split_idx:]
testY = Y[split_idx:]

#%%
reg_model = LinearRegression()
reg_model.fit(trainX, trainY)
print(reg_model.score(trainX, trainY))

predY = reg_model.predict(testX)
np.diag(cosine_similarity(testY, predY))

# %%
