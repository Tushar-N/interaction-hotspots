import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import glob
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance
import cv2

from utils import util
import data
from data import opra, epic
from models import rnn, backbones, intcam

cudnn.benchmark = True 
parser = argparse.ArgumentParser()
parser.add_argument('--dset', default=None)
parser.add_argument('--load', default=None)
parser.add_argument('--inp', default=None)
parser.add_argument('--out', default=None)
args = parser.parse_args()


def resize_tensor(tensor, sz):
    tensor = F.interpolate(tensor, (sz, sz), mode='bilinear', align_corners=True)
    return tensor

def blur(tensor, sz, Z): # (3, 224, 224)
    tensor = tensor.permute(1,2,0).numpy()
    k_size = int(np.sqrt(sz**2) / Z)
    if k_size % 2 == 0:
        k_size += 1
    tensor = cv2.GaussianBlur(tensor, (k_size, k_size), 0)
    tensor = torch.from_numpy(tensor).permute(2,0,1)
    return tensor

def post_process(hmaps, sz):
    hmaps = torch.stack([hmap/(hmap.max() + 1e-12) for hmap in hmaps], 0)
    hmaps = hmaps.numpy()

    processed = []        
    for c in range(hmaps.shape[0]):
        hmap = hmaps[c]
        hmap[hmap<0.5] = 0
        hmap = cv2.GaussianBlur(hmap, (3, 3), 0)
        processed.append(hmap)
    processed = np.array(processed)
    processed = torch.from_numpy(processed).float()
    processed = resize_tensor(processed.unsqueeze(0), sz)[0]

    return processed

color_map = {'red':[1,0,0], 'green':[0,1,0], 'blue':[0,0,1],
            'cyan':[0,1,1], 'magenta':[1,0,1], 'yellow':[1,1,0]}
def generate_color_map(hmaps, colors, sz):
    colors = [color_map[c] for c in colors]
    colors = 1 - torch.FloatTensor(colors).unsqueeze(2).unsqueeze(2) # invert colors

    vals, idx = torch.sort(hmaps, 0, descending=True)
    cmap = torch.zeros(hmaps.shape)
    for c in range(hmaps.shape[0]):
        cmap[c][idx[0]==c] = vals[0][idx[0]==c]

    cmap = cmap.unsqueeze(1).expand(cmap.shape[0], 3, cmap.shape[-1], cmap.shape[-1]) # (C, 3, 224, 224)
    cmap = [hmap*color for hmap, color in zip(cmap, colors)]
    cmap = torch.stack(cmap, 0) # (C, 3, 14, 14)

    cmap = resize_tensor(cmap, sz)
    cmap, _ = cmap.max(0)

    # blur the heatmap to make it smooth
    cmap = blur(cmap, sz, 9)
    cmap = 1 - cmap # invert heatmap: white background

    # improve contrast for visibility
    cmap = transforms.ToPILImage()(cmap)
    cmap = ImageEnhance.Color(cmap).enhance(1.5)
    cmap = ImageEnhance.Contrast(cmap).enhance(1.5)
    cmap = transforms.ToTensor()(cmap)

    return cmap

def overlay_colored_heatmaps(uimg, hmaps, viz_idx, colors, sz): # (C, 224, 224)

    # post process heatmaps: normalize each channel, blur, threshold
    hmaps = post_process(hmaps, sz) # (C, 224, 224)
    hmaps = hmaps[viz_idx] 

    # generate color map from each heatmap channel
    cmap = generate_color_map(hmaps, colors, sz)

    # generate per-pixel alpha channel and overlay
    alpha = (1-cmap).mean(0)
    overlay = (1-alpha)*uimg + alpha*cmap

    return overlay

def visualize(path, viz_verbs, colors, sz=256):

    # load image
    img = Image.open(path).convert('RGB')
    img = util.default_transform('val')(img)

    # generate heatmaps
    hmaps = gcam.generate_cams(img.cuda().unsqueeze(0), list(range(len(dataset.verbs)))) # (1, T, C, 28, 28)
    hmaps = hmaps.mean(1).squeeze(0).cpu() # (C, 28, 28)

    # overlay heatmaps on original image
    uimg = util.unnormalize(img)
    uimg = F.interpolate(uimg.unsqueeze(0), (sz, sz), mode='bilinear', align_corners=False)[0]

    viz_idx = [dataset.verbs.index(v) for v in viz_verbs]
    overlay = overlay_colored_heatmaps(uimg, hmaps, viz_idx, colors, sz)

    # display heatmaps next to original
    viz_imgs = [uimg, overlay]
    grid = torchvision.utils.make_grid(viz_imgs, nrow=1, padding=2)
    grid = transforms.ToPILImage()(grid)
    return grid

#-----------------------------------------------------------------------------------------------------#

if args.dset=='opra':
    dataset = opra.OPRAHeatmaps(root=data._DATA_ROOTS['opra'], split='val')
    viz_verbs = ['hold', 'rotate', 'push']
    colors = ['red', 'green', 'blue']
elif args.dset=='epic':
    dataset = epic.EPICHeatmaps(root=data._DATA_ROOTS['epic'], split='val')
    viz_verbs = ['cut', 'mix', 'open', 'close']
    colors = ['red', 'green', 'cyan', 'cyan']

torch.backends.cudnn.enabled = False
net = rnn.frame_lstm(len(dataset.verbs), max_len=-1, backbone=backbones.dr50_n28)

checkpoint = torch.load(args.load, map_location='cpu')
net.load_state_dict(checkpoint['net'])
print ('Loaded checkpoint from %s'%os.path.basename(args.load))

gcam = intcam.IntCAM(net)
gcam.eval().cuda()

os.makedirs(args.out, exist_ok=True)

for fl in glob.glob('%s/*.jpg'%args.inp):
    img = visualize(fl, viz_verbs, colors)
    img.save('%s/%s'%(args.out, os.path.basename(fl)))
    print (fl)