import os
import glob
import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root', help='directory which OPRA has been downloaded to')
args = parser.parse_args()

os.makedirs(args.root + '/data/frames/', exist_ok=True)
for fl in tqdm.tqdm(glob.glob(args.root + '/data/clips/*/*/*/*.mp4')):
    _, _, channel, playlist, product, clip = fl.split('/')[-6:] # data/clips/seattle/1/76/428.0_3.0.mp4
    out_dir = args.root + '/data/frames/%s/%s/%s/%s/'%(channel, playlist, product, clip)
    os.makedirs(out_dir, exist_ok=True)
    cmd = 'ffmpeg -i %s -vf "scale=256:256,fps=5" %s/image-%%08d.jpg'%(fl, out_dir)
    os.system(cmd)