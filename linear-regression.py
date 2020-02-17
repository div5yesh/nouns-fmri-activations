# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
from itertools import groupby, combinations
import numpy as np
import scipy.io
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

# %%
def load_subject_data(idx):
    mat = scipy.io.loadmat('data-science-P' + str(idx) + '.mat')
    meta = mat["meta"]
    info = mat["info"]
    data = mat["data"]
    # voxel_map = meta["coordToCol"][0][0]    #(51, 61, 23)
    # voxel_vec = meta["colToCoord"][0][0]    #(21764, 3)
    # scan_shape = voxel_map.shape
    return meta, info, data

def prepare_image(scan_shape, data, voxel_map):
    fmri_image = np.zeros(scan_shape)
    for i in range(scan_shape[0]):
        for j in range(scan_shape[1]):
            for k in range(scan_shape[2]):
                voxel = voxel_map[i][j][k]
                fmri_image[i][j][k] = data[voxel - 1]
    return fmri_image

def show_slices(fmri_image):
    for i in range(scan_shape[2]):
        plt.imshow(fmri_image[:,:,i])
        plt.show()

def get_trial_info(info):
    wrd_to_trial = dict()
    trial_count = len(info[0])
    trial_to_wrd = []
    for i in range(len(info[0])):
        trial = info[0][i]
        condition_idx = trial['cond_number'][0][0]
        word_idx = trial['word_number'][0][0]
        word = trial['word'][0]
        key = (condition_idx, word_idx, word)
        wrd_to_trial[word] = wrd_to_trial.get(word, []) + [i]
        trial_to_wrd += [word]
    return trial_to_wrd, wrd_to_trial

def get_semantic_features():
    fp = open("features.txt", "r")
    feat_data = fp.read()
    fp.close()

    features = dict()
    words = feat_data.split("\n\n\n")
    for word in words:
        feature = word.split("\n\n")
        value = list(map(lambda x:float(x.split(" ")[1]), sorted(feature[1].split("\n"), key=lambda x: x[0])))
        features[feature[0]] = value

    return features

def plot_slices(fmri_image):
    fig, ax = plt.subplots(nrows=11, ncols=2, figsize=(50,100))
    idx = 0
    for row in ax:
        for col in row:
            col.imshow(fmri_image[:,:,idx])
            idx += 1
    plt.tight_layout()
    plt.show()

def normalize(data):
    samples = len(data)
    images = []
    for i in range(samples):
        images += [data.flatten()[i].flatten()]

    images = np.array(images)
    # mean_image = np.mean(images, axis=0)
    # images = images - mean_image
    return images

def selection(image):
    return image[:]

def match(pred, act):
    c1 = cosine_similarity(selection(pred[0]), selection(act[1]))
    c2 = cosine_similarity(selection(pred[1]), selection(act[0]))
    return c1 + c2

# %%
p1_meta, p1_info, p1_data = load_subject_data(1)
p1_data_flat = normalize(p1_data)
trial_info, trial_map = get_trial_info(p1_info)
features = get_semantic_features()
# X, Y = prepare_data(p1_data, features, trial_info)

# %%
nouns = set(trial_map.keys())
test_comb = combinations(nouns, 2)
iteration = 0
for test_nouns in test_comb:
    train_nouns = nouns - set(test_nouns)

    train_features = []
    image_reps = []
    for nn in train_nouns:
        train_features += [features[nn]]
        image_ids = trial_map[nn]
        images = []
        for i in image_ids:
            img = p1_data_flat[i]
            images += [img]

        images = np.array(images)
        image_rep = np.mean(images, axis=0)
        image_reps += [image_rep]

    image_reps = np.array(image_reps)
    mean = np.mean(image_reps, axis=0)

    train_features = np.array(train_features)   # 58, 25
    train_images = image_reps - mean

    reg_model = LinearRegression()
    reg_model.fit(train_features, train_images)
    
    # TODO: voxel selection using voxel_map = (21764, 6, 58)


    iteration += 1
    break

print(iteration)    #1770

# %%
p1_voxel_map = p1_meta["coordToCol"][0][0]
scan_shape = p1_voxel_map.shape

sample_image = prepare_image(scan_shape, predY[0], p1_voxel_map)
plot_slices(sample_image)

# %%
sample_image = prepare_image(scan_shape, testY[0], p1_voxel_map)
plot_slices(sample_image)

# %%
