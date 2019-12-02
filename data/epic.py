import collections
import os
from PIL import Image
import tqdm
import numpy as np
import json

from utils import util
from data.hotspot_dataset import VideoInteractions, HeatmapDataset
from data.hotspot_dataset import generate_heatmaps

#---------------------------------------------------------------------------------------------------#

class ImageLoader:
    def __init__(self, root, rgb_or_det):
        if rgb_or_det=='rgb':
            self.frame_dir = '%s/frames_rgb_flow/rgb/train/'%(root)
            self.prefix = 'frame_'
        elif rgb_or_det=='det':
            self.frame_dir = '%s/object_detection_images/train/'%(root)
            self.prefix = ''

    def __call__(self, v_id, f_id):
        file = self.frame_dir + '/%s/%s/%s%010d.jpg'%(v_id.split('_')[0], v_id, self.prefix, f_id)
        img = Image.open(file).convert('RGB')
        return img


def expand_box(img, box, expand):
    # add a little padding to the box
    width, height = box[2]-box[0], box[3]-box[1]
    delta_x, delta_y = int(width*expand), int(height*expand)

    W, H = img.size
    xmin, ymin, xmax, ymax = box
    if delta_x>0:
        pad_x_left = min(delta_x//2, xmin)
        pad_x_right = min(delta_x//2, W-xmax)
    if delta_y>0:
        pad_y_top = min(delta_y//2, ymin)
        pad_y_bot = min(delta_y//2, H-ymax)

    W, H = img.size
    new_box = [max(xmin-pad_x_left, 0), max(ymin-pad_y_top, 0),
               min(xmax+pad_x_right, W-1), min(ymax+pad_y_bot, H-1)]

    return new_box

class CropPerturb:

    def __init__(self, root):
        self.loader = ImageLoader(root, 'det')

    def __call__(self, entry):
        v_id, f_id, box = entry['v_id'], entry['f_id'], entry['box']

        det_img = self.loader(v_id, f_id)

        expand = np.random.uniform(0.5, 1.5)
        xmin, ymin, xmax, ymax = expand_box(det_img, box, expand)
        box_W = xmax-xmin
        box_H = ymax-ymin

        tx = np.random.uniform(-0.2, 0.2)
        ty = np.random.uniform(-0.2, 0.2)
        offset_x, offset_y = tx*box_W, ty*box_H

        _xmin = xmin + offset_x
        _ymin = ymin + offset_y
        _xmax = xmax + offset_x
        _ymax = ymax + offset_y

        box = list(map(int, [_xmin, _ymin, _xmax, _ymax]))

        crop = det_img.crop(box)

        return crop

class EPICInteractions(VideoInteractions):

    def __init__(self, root, split, max_len, sample_rate=10):
        super().__init__(root, split, max_len, sample_rate)

        annots = json.load(open('data/epic/annotations.json'))
        self.verbs, self.nouns = annots['verbs'], annots['nouns']
        self.train_data, self.val_data = annots['train_clips'], annots['test_clips']
        self.data = self.train_data if self.split=='train' else self.val_data
        print ('Train data: %d | Val data: %d'%(len(self.train_data), len(self.val_data)))

        # Use every frame. For EPIC sample_rate = 10 --> 6fps
        for entry in self.train_data + self.val_data:
            entry['frames'] = [(entry['v_id'], f_id) for f_id in range(entry['start'], entry['stop']+1, self.sample_rate)]

        self.rgb_loader = ImageLoader(self.root, 'rgb')
        self.box_cropper = CropPerturb(self.root)

        self.inactive_images = annots['inactive_images']

    def load_frame(self, frame):
        v_id, f_id = frame
        return self.rgb_loader(v_id, f_id)

    def select_inactive_instances(self, entry):

        def select(noun):
            candidates = self.inactive_images[noun]
            img = candidates[np.random.randint(len(candidates))]
            crop = self.box_cropper(img)
            crop = self.img_transform(crop)
            return crop

        pos_noun = self.nouns[entry['noun']]

        candidate_nouns = list(self.inactive_images.keys())
        neg_noun = candidate_nouns[np.random.randint(len(candidate_nouns))]

        positive, negative = select(pos_noun), select(neg_noun)
        return positive, negative

    def __len__(self):
        return len(self.data)

#---------------------------------------------------------------------------------#

class EPICHeatmaps(HeatmapDataset):
    def __init__(self, root, split, std_norm=True):
        hm_file = 'data/epic/heatmaps.h5'
        super().__init__(root, split, hm_file=hm_file, std_norm=std_norm)

        annots = json.load(open('data/epic/annotations.json'))
        if not os.path.exists(hm_file):
            generate_heatmaps(annots, kernel_size=3.0, out_file=hm_file, transpose=True)

        self.verbs = annots['verbs']
        self.train_data, self.val_data = annots['train_images'], annots['test_images']
        self.data = self.train_data if self.split=='train' else self.val_data
        print ('%d train images, %d test images'%(len(self.train_data), len(self.val_data)))     

    def load_image_heatmap(self, entry):
        path = 'data/epic/images/%s'%(entry['image'][0]) # P01_17_301_1084_92_1412_462 ...
        crop = util.load_img(path)

        hm_key = tuple(entry['image']) + (str(entry['verb']), )
        heatmap = self.heatmaps(hm_key)

        crop, heatmap = self.pair_transform(crop, heatmap)

        return crop, heatmap

#---------------------------------------------------------------------------------#