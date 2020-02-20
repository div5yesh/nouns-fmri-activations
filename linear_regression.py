# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
from itertools import groupby, combinations
import numpy as np
import scipy.io
from scipy.stats.stats import pearsonr
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

def show_slices(fmri_image, scan_shape=(51, 61, 23)):
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

def flatten(data):
    samples = len(data)
    images = []
    for i in range(samples):
        images += [data.flatten()[i].flatten()]

    images = np.array(images)
    return images

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
    c1 = similarity[0][1]
    c2 = similarity[1][0]
    return c1 + c2

def prepare_data(features, trial_map, data_flat, stimuli):
    semantic_embeds = []
    image_representatives = []
    voxel_map = []
    for noun in stimuli:
        semantic_embeds += [features[noun]]
        image_ids = trial_map[noun]
        sample_images = []
        for i in image_ids:
            image = data_flat[i]
            sample_images += [image]

        voxel_map += [sample_images]
        sample_images = np.array(sample_images)
        representative = np.mean(sample_images, axis=0)
        image_representatives += [representative]

    voxel_map = np.array(voxel_map).T
    image_representatives = np.array(image_representatives)
    mean = np.mean(image_representatives, axis=0)

    semantic_embeds = np.array(semantic_embeds)   # (N, 25)
    fmri_images = image_representatives - mean  # (N, 21764)
    return fmri_images, semantic_embeds, voxel_map

# %%
p1_meta, p1_info, p1_data = load_subject_data(1)
p1_data_flat = flatten(p1_data)
trial_info, trial_map = get_trial_info(p1_info)
features = get_semantic_features()

# %%
def train(features, trial_map, data_flat):
    nouns = set(trial_map.keys())
    test_combinations = combinations(nouns, 2)
    iteration = 0
    predictions = []
    true_images = []
    training_voxel_map = []
    for test_nouns in test_combinations:
        train_nouns = nouns - set(test_nouns)

        # prepare dataset
        train_images, train_features, voxel_map = prepare_data(features, trial_map, p1_data_flat, train_nouns)
        test_images, test_features, _ = prepare_data(features, trial_map, p1_data_flat, test_nouns)

        training_voxel_map += [voxel_map]

        # model
        reg_model = LinearRegression()
        reg_model.fit(train_features, train_images)
        pred_images = reg_model.predict(test_features)

        predictions += [pred_images]
        true_images += [test_images]

        iteration += 1
        print("Training Combination: %d/%d" % (iteration, 1770))

        if iteration == 100:
            break

    print("Done")    #1770
    return training_voxel_map, predictions, true_images

def evaluate(training_voxel_map, predictions, true_images):
    similarity_map = []
    traditional = []
    for i in range(len(predictions)):
        top500voxels = find_top500voxels(training_voxel_map[i])

        predicted_voxels = select(top500voxels, predictions[i])
        actual_voxels = select(top500voxels, true_images[i])

        similarity = match(predicted_voxels, actual_voxels)
        similarity_map += [similarity]
        
        similarity = cosine_similarity(predicted_voxels, actual_voxels)
        traditional += [similarity[0][0] + similarity[1][1]]

        print('Eval Combination: %d' % (i))

    return similarity_map, traditional

# %%
training_voxel_map, predictions, true_images = train(features, trial_map, p1_data_flat)

#%%
similarity, traditional = evaluate(training_voxel_map, predictions, true_images)

#%%
p1_voxel_map = p1_meta["coordToCol"][0][0]
scan_shape = p1_voxel_map.shape

sample_image = prepare_image(scan_shape, predictions[0][0], p1_voxel_map)
plot_slices(sample_image)

# %%
sample_image = prepare_image(scan_shape, true_images[0][0], p1_voxel_map)
plot_slices(sample_image)

# %%
