import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import tqdm
import torchnet as tnt
import collections
import logging

from utils import util

cudnn.benchmark = True 
parser = argparse.ArgumentParser()
parser.add_argument('--dset', default=None, help='opra|epic')
parser.add_argument('--max_len', default=8, type=int, help='Length of frame sequence input to LSTM')
parser.add_argument('--cv_dir', default='cv/tmp/', help='Directory for saving checkpoint models')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--max_epochs', default=20, type=int, help='Total number of training epochs')
parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--decay_after', default=15, type=float, help='Epoch for scheduler to decay lr by 10x')
parser.add_argument('--parallel', action ='store_true', default=False, help='Use nn.DataParallel')
parser.add_argument('--workers', type=int, default=8, help='Workers for dataloader')
parser.add_argument('--log_every', default=10, type=int, help='Logging frequency')
args = parser.parse_args()

os.makedirs(args.cv_dir, exist_ok=True)
logging.basicConfig(filename='%s/run.log'%args.cv_dir, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

def save(epoch):
    logger.info('Saving state, epoch: %d'%epoch)
    state_dict = net.state_dict() if not args.parallel else net.module.state_dict()
    checkpoint = {'net':state_dict, 'args':args, 'epoch': epoch}
    torch.save(checkpoint, '%s/ckpt_E_%d.pth'%(args.cv_dir, epoch))

def train(epoch):

    net.train()

    iteration = 0
    total_iters = len(trainloader)
    loss_meters = collections.defaultdict(lambda: tnt.meter.MovingAverageValueMeter(20))
    for batch in trainloader:

        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

        loss_dict = {k:v.mean() for k,v in loss_dict.items()}
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred_idx = pred.max(1)
        correct = (pred_idx==batch['verb']).float().sum()
        batch_acc = correct/pred.shape[0]
        loss_meters['bacc %'].add(batch_acc.item())

        for k, v in loss_dict.items():
            loss_meters[k].add(v.item())
        loss_meters['total_loss'].add(loss.item())

        if iteration%args.log_every==0:
            log_str = 'epoch: %d + %d/%d | '%(epoch, iteration, total_iters)
            log_str += ' | '.join(['%s: %.3f'%(k, v.value()[0]) for k,v in loss_meters.items()])
            logger.info(log_str)

        iteration += 1

#----------------------------------------------------------------------------------------------------------------------------------------#
import data
from data import opra, epic

if args.dset=='opra':
    trainset = opra.OPRAInteractions(root=data._DATA_ROOTS[args.dset], split='train', max_len=args.max_len)
    ant_loss = 'mse'
elif args.dset=='epic':
    trainset = epic.EPICInteractions(root=data._DATA_ROOTS[args.dset], split='train', max_len=args.max_len)
    ant_loss = 'triplet'

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, sampler=trainset.data_sampler())

from models import rnn, backbones
torch.backends.cudnn.enabled = False
net = rnn.frame_lstm(len(trainset.verbs), trainset.max_len, backbone=backbones.dr50_n28, ant_loss=ant_loss)
net.cuda()

if args.parallel:
    net = nn.DataParallel(net)

optim_params = list(filter(lambda p: p.requires_grad, net.parameters()))
logger.info('# params to optimize %s'%len(optim_params))
optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.decay_after], gamma=0.1)

start_epoch = 1 # or load checkpoint
for epoch in range(start_epoch, args.max_epochs+1):
    logger.info('LR = %.2E'%scheduler.get_lr()[0])

    train(epoch)
    scheduler.step()

    if epoch==args.max_epochs:
        save(epoch)
