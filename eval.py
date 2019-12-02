import torch
import argparse
import tqdm
import os 
import glob

from utils import evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--dset', default='opra')
parser.add_argument('--load', default=None)
parser.add_argument('--res', type=int, default=28)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()
#------------------------------------------------------------#

import data
from data import opra, epic
import torch.nn.functional as F
def generate_gt(dset):

    os.makedirs('data/%s/output/'%dset, exist_ok=True)

    if dset=='opra':
        dataset = opra.OPRAHeatmaps(root=data._DATA_ROOTS[dset], split='val') 
    elif dset=='epic':
        dataset = epic.EPICHeatmaps(root=data._DATA_ROOTS[dset], split='val') 

    dataset.heatmaps = dataset.init_hm_loader()    

    heatmaps, keys = [], []
    for index in tqdm.tqdm(range(len(dataset))):
        entry = dataset.data[index]
        hm_key = tuple(entry['image']) + (str(entry['verb']),)
        heatmap = dataset.heatmaps(hm_key)
        heatmap = torch.from_numpy(heatmap)
        heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0][0]
        heatmap = heatmap/(heatmap.sum()+1e-12)

        heatmaps.append(heatmap)
        keys.append(hm_key)

    heatmaps = torch.stack(heatmaps, 0)
    print (heatmaps.shape)
    torch.save({'heatmaps':heatmaps, 'keys':keys}, 'data/%s/output/gt.pth'%(dset))

#------------------------------------------------------------#

from models import intcam
def generate_heatmaps(dset, load, batch_size):

    if dset=='opra':
        testset = opra.OPRAHeatmaps(root=data._DATA_ROOTS[dset], split='val') 
    elif dset=='epic':
        testset = epic.EPICHeatmaps(root=data._DATA_ROOTS[dset], split='val') 

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    from models import rnn, backbones
    torch.backends.cudnn.enabled = False
    net = rnn.frame_lstm(len(testset.verbs), max_len=-1, backbone=backbones.dr50_n28)

    checkpoint = torch.load(load, map_location='cpu')
    weights = checkpoint['net']
    net.load_state_dict(weights)
    net.eval().cuda()
    print ('Loaded checkpoint from %s'%os.path.basename(load))
    
    gcam = intcam.IntCAM(net)

    heatmaps = []
    for batch in tqdm.tqdm(testloader, total=len(testloader)):
    
        img, verb = batch['img'], batch['verb'] 
        masks = gcam.generate_cams(img.cuda(), [verb]) # (B, T, C, 7, 7)
        mask = masks.mean(1) # (B, C, 7, 7) <-- average across hallucinated time dim
        mask = mask.squeeze(1) # get rid of single class dim
        heatmaps.append(mask.cpu())

    heatmaps = torch.cat(heatmaps, 0) # (N, C, 7, 7)
    print (heatmaps.shape)

    keys = [testset.key(entry) for entry in testset.data]
    torch.save({'heatmaps':heatmaps, 'keys':keys}, '%s.%s.heatmaps'%(load, dset))


#------------------------------------------------------------#

if __name__=='__main__':

    # generate gt heatmaps if they do not already exist
    if not os.path.exists('data/%s/output/gt.pth'%(args.dset)):
        generate_gt()

    # generate heatmap predictions if they do not already exist
    if args.load and not os.path.exists('%s.%s.heatmaps'%(args.load, args.dset)):
        generate_heatmaps(args.dset, args.load, args.batch_size)
    print ('loading checkpoint:', args.load)

    gt = torch.load('data/%s/output/gt.pth'%(args.dset))
    baselines = evaluation.Baselines(gt['heatmaps'].shape[0])
    heval = evaluation.Evaluator(gt, res=args.res, log=args.load)

    # Comment in other methods to compare
    predictions = {
        # 'center': baselines.gaussian(),
        # 'egogaze': baselines.checkpoint('data/%s/output/egogaze.pth'%(args.dset)),
        # 'mlnet': baselines.checkpoint('data/%s/output/mlnet.pth'%(args.dset)),
        # 'deepgaze2': baselines.checkpoint('data/%s/output/deepgaze2.pth'%(args.dset)),
        # 'salgan': baselines.checkpoint('data/%s/output/salgan.pth'%(args.dset)),
        'hotspots': baselines.checkpoint('%s.%s.heatmaps'%(args.load, args.dset)),
        # 'img2heatmap': baselines.checkpoint('data/%s/output/img2heatmap.pth'%(args.dset)),
        }
    if args.dset=='opra':
        predictions.update({
            # 'demo2vec': baselines.checkpoint('data/opra/output/d2v.pth'),
            })

    for method in predictions:
        print (method)
        heatmaps = predictions[method]
        scores = heval.evaluate(heatmaps)

