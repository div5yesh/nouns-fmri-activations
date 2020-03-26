#%%
import numpy as np
import scipy.io, pickle, os

path = os.path.join(os.getcwd(), 'mitchell')

class Data:
    def __init__(self, samples, voxel_map, trial_map):
        self.samples = samples
        self.voxel_map = voxel_map
        self.trial_map = trial_map

class DataLoader:
    def __init__(self):
        self.data = self.load()
        self.features = self.get_semantic_features()

    def load(self):
        dataset = {}
        for idx in range(1,10):
            dataset[idx] = self.load_subject_data(idx)
        return dataset

    def load_subject_data(self, idx):
        mat = scipy.io.loadmat(os.path.join(path, 'data-science-P%d.mat' % (idx)))
        subject_meta = mat["meta"]
        subject_info = mat["info"]
        subject_data = mat["data"]

        voxel_map = (subject_meta["colToCoord"][0][0],subject_meta["coordToCol"][0][0])
        samples = self.flatten(subject_data)
        trial_map = self.get_trial_info(subject_info)

        data = Data(samples, voxel_map, trial_map)
        return data

    def flatten(self, data):
        samples = len(data)
        images = []
        for i in range(samples):
            images += [data.flatten()[i].flatten()]

        images = np.array(images)
        return images

    def get_trial_info(self, info):
        word2trial = dict()
        for i in range(len(info[0])):
            trial = info[0][i]
            word = trial['word'][0]
            word2trial[word] = word2trial.get(word, []) + [i]
        return word2trial

    def get_semantic_features(self):
        fp = open(os.path.join(path, 'features.txt'), "r")
        feat_data = fp.read()
        fp.close()

        features = dict()
        words = feat_data.split("\n\n\n")
        for word in words:
            feature = word.split("\n\n")
            value = list(map(lambda x:float(x.split(" ")[1]), sorted(feature[1].split("\n"), key=lambda x: x[0])))
            features[feature[0]] = value

        return features

class Preprocessor:
    def get_snr(self, idx, samples, trial_map):
        snr = None
        snr_path = os.path.join(os.getcwd(),'snr','snr%d.h5' % (idx))
        print(snr_path)
        if not os.path.isfile(snr_path):
            groups = self.get_grouped_samples(trial_map, samples)
            snr = self.calculate_snr(groups.T)
            pickle.dump(snr, open(snr_path, "wb"))
        else:
            snr = pickle.load(open(snr_path, "rb"))
        return snr

    def prepare_data(self, features, trial_map, data_flat, stimuli):
        semantic_embeds = []
        image_representatives = []
        
        for noun in stimuli:
            semantic_embeds += [features[noun]]
            image_ids = trial_map[noun]
            sample_images = data_flat[image_ids]
            representative = np.mean(sample_images, axis=0)
            image_representatives += [representative]

        image_representatives = np.array(image_representatives)
        mean = np.mean(image_representatives, axis=0)

        semantic_embeds = np.array(semantic_embeds)   # (N, 25)
        fmri_images = image_representatives - mean  # (N, 21764)
        return fmri_images, semantic_embeds

    def get_grouped_samples(self, trial_map, samples):
        sample_group = []
        for noun in trial_map.keys():
            image_ids = trial_map[noun]
            sample_images = samples[image_ids]
            sample_group += [sample_images]

        sample_group = np.array(sample_group) # (60, 6)
        return sample_group

    def calculate_snr(self, sample_group):
        error_v_i_l = np.zeros(sample_group.shape)
        error_std = np.zeros(len(sample_group))

        singal_v_l = np.mean(sample_group,axis=1)

        for v in range(sample_group.shape[0]):
            for l in range(60):
                for i in range(6):
                    error_v_i_l[v][i][l] = sample_group[v][i][l] - singal_v_l[v][l]

        error_mean = np.mean(np.mean(error_v_i_l, axis=2),axis=1)

        for v in range(sample_group.shape[0]):
            error_std[v] = np.sqrt(np.mean((error_v_i_l[v] - error_mean[v]) ** 2))

        signal_std = np.std(singal_v_l, axis=1)

        snr = signal_std/error_std
        return snr

preprocess = Preprocessor()
dataset = DataLoader()

# %%
