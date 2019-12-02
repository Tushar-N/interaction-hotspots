import torch.utils.data as tdata
import collections
import torch
import numpy as np
import h5py
from joblib import Parallel, delayed

from utils import util


class VideoInteractions(tdata.Dataset):

    def __init__(self, root, split, max_len, sample_rate):
        self.root = root
        self.split = split
        self.max_len = max_len
        self.sample_rate = sample_rate

        if self.max_len==-1:
            self.max_len = 32
            print ('Max length not chosen. Setting max length to:', self.max_len)
        
        self.clip_transform = util.clip_transform(self.split, self.max_len)
        self.img_transform = util.default_transform(self.split)
        self.pair_transform = util.PairedTransform(self.split)
    
    # Function to load each frame in entry['frames']
    # return: PIL image for frame
    def load_frame(self, frame):
        pass

    # Function to select the positive (and optional negative) inactive image for L_{ant}
    # return: positive (3,224,224), negative (3,224,224)
    def select_inactive_instances(self, entry):
        pass

    # Weighted sampler for class imbalance
    def data_sampler(self):
        counts = collections.Counter([entry['verb'] for entry in self.data])
        icounts = {k:sum([counts[v] for v in counts])/counts[k] for k in counts}
        icounts = {k:min(icounts[k], 100) for k in icounts}
        weights = [icounts[entry['verb']] for entry in self.data]
        weights = np.array(weights)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        return sampler

    # sample a randomly placed window of self.max_len frames from the video clip
    def sample(self, clip):

        if len(clip)<=self.max_len:
            return clip

        if self.split=='train':
            st = np.random.randint(0, len(clip)-self.max_len)
        elif self.split=='val':
            st = len(clip)//2 - self.max_len//2
        clip = clip[st:st+self.max_len]

        return clip

    def __getitem__(self, index):

        entry = self.data[index]

        #--------------------------------------------------------------------------#
        # sample frames and load/transform
        frames = self.sample(entry['frames'])
        length = len(frames)
        frames = [self.load_frame(frame) for frame in frames]
        frames = self.clip_transform(frames) # (T, 3, 224, 224)
        instance = {'frames':frames, 'verb':entry['verb'], 'noun':entry['noun'], 'length':length}

        #--------------------------------------------------------------------------#
        # load the positive and negative images for the triplet loss
        positive, negative = self.select_inactive_instances(entry)
        instance.update({'positive':positive, 'negative':negative})

        #--------------------------------------------------------------------------#

        return instance

    def __len__(self):
        return len(self.data)


#----------------------------------------------------------------------------------#

import cv2
def compute_heatmap(points, image_size, k_ratio, transpose):
    """Compute the heatmap from annotated points.
    Args:
        points: The annotated points.
        image_size: The size of the image.
        k_ratio: The kernal size of Gaussian blur.
    Returns:
        The heatmap array.
    """
    points = np.asarray(points)
    heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
    n_points = points.shape[0]

    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        row = int(x)
        col = int(y)

        try:
            heatmap[row, col] += 1.0
        except:
            # resize pushed it out of bounds somehow
            row = min(max(row, 0), image_size[0]-1)
            col = min(max(col, 0), image_size[1]-1)
            heatmap[row, col] += 1.0
        
    # Compute kernel size of the Gaussian filter. The kernel size must be odd.
    k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1

    # Compute the heatmap using the Gaussian filter.
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)

    if np.sum(heatmap)>0:
        heatmap /= np.sum(heatmap)

    if transpose:
        heatmap = heatmap.transpose()

    return heatmap    

def generate_heatmaps(annots, kernel_size, out_file, transpose):

    def generate(images):
        print ('Generating %d heatmaps'%len(images))
        keys = [tuple(entry['image']) + (str(entry['verb']), ) for entry in images]
        hmaps = Parallel(n_jobs=16, verbose=2)(delayed(compute_heatmap)(entry['points'], entry['shape'], kernel_size, transpose) for entry in images)
        return keys, hmaps

    train_keys, train_hmaps = generate(annots['train_images'])
    test_keys, test_hmaps = generate(annots['test_images'])

    # save the heatmaps as an h5 file
    hf = h5py.File(out_file, 'w')
    keys = [np.array(key, dtype='S') for key in train_keys+test_keys]
    hf.create_dataset('keys', data=keys, dtype=h5py.special_dtype(vlen=str))
    for idx, hmap in enumerate(train_hmaps+test_hmaps):
        hf.create_dataset('heatmaps/%d'%idx, data=hmap, dtype=np.float32)
    hf.close()


#----------------------------------------------------------------------------------#

class HeatmapLoader:
    def __init__(self, hf):
        hf = h5py.File(hf, 'r')
        self.heatmaps = hf['heatmaps']
        self.map = {tuple(k): str(idx) for idx, k in enumerate(np.array(hf['keys']))}

    def __call__(self, key):
        heatmap = self.heatmaps[self.map[key]]
        heatmap = np.array(heatmap)
        return heatmap

class HeatmapDataset(tdata.Dataset):

    def __init__(self, root, split, hm_file, std_norm=True):
        self.root = root
        self.split = split
        self.heatmaps = None
        self.hm_file = hm_file
        self.pair_transform = util.PairedTransform(self.split, std_norm)

    def init_hm_loader(self):
        return HeatmapLoader(self.hm_file)

    # Function to load the inactive image + its associated heatmap
    def load_image_heatmap(self, entry):
        pass

    # return the key for the entry (used for matching .h5 heatmaps)
    def key(self, entry):
        return tuple(entry['image']) + (str(entry['verb']), )
  
    def __getitem__(self, index):

        if self.heatmaps is None:
            self.heatmaps = self.init_hm_loader()

        entry = self.data[index]
        img, heatmap = self.load_image_heatmap(entry)
        instance = {'img':img, 'verb':entry['verb'], 'heatmap':heatmap}
        return instance

    def __len__(self):
        return len(self.data)
