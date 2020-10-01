import os
from os import path
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import pickle
import numpy as np
from tempfile import TemporaryFile
from tqdm import tqdm
import argparse
import logging
import time
import pandas as pd

import torch
from torch.nn import functional as F
from torch.nn.functional import conv2d
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torchvision.models as models
from torch import nn

from functions.cmc import compute_cmc
from functions.normcorr import patch_mean, patch_std, channel_normalize, NCC, calc_corr

folder = 'C:\\Users\\User\\virtualtest\\MCNCC2\\datasets\\FID-300'

parser = argparse.ArgumentParser(description='take some indiviudal folders from user')
parser.add_argument('-t', '--tracks', type=str, default='tracks_cropped_Subset', help='define track folder')
parser.add_argument('-rf', '--refs', type=str, default='Subset', help='define reference folder')
parser.add_argument('-str', '--stride', type=int, default='1', help='stride for convolutions')
parser.add_argument('-avgp', '--avgpool_bool', default='False', help='activate average pooling for features')
parser.add_argument('-r', '--rot', default=False, action='store_true', help='add rotation')
parser.add_argument('-ris', '--start', type=int, default=-10, help='rotation interval start')
parser.add_argument('-rie', '--end', type=int, default=11, help='rotation interval end')
parser.add_argument('-sf', '--scorefile', type=str, default='scores.npy', help='scorefilename')
parser.add_argument('-cmc', '--cmc', default=False, action='store_true', help='calculate cmc')
parser.add_argument('-cmcf', '--cmc_file', type=str, default='cmc_file', help='cmc filename')
parser.add_argument('-lbltable', '--label_file', type=str, default='Subsetlabels.csv', help='name of the csv. file')

args = parser.parse_args()

tracks = path.join(folder, args.tracks)
refs = path.join(folder, args.refs)

ref_l = [f for f in os.listdir(refs) if f.endswith('.png')]
track_l = [f for f in os.listdir(tracks) if f.endswith('.jpg')]

if __name__ == "__main__":
    print("refs:", args.refs)
    print("tracks:", args.tracks)
    print("stride:", args.stride)
    print("rot:", args.rot)
    print("start:", args.start)
    print("end:", args.end)
    print("cmc:", args.cmc)
    print("scorefile:", args.scorefile)
    print("cmc_file:", args.cmc_file)
    print("label_file:", args.label_file)
    print("average_pooling:", args.avgpool_bool)

device = torch.device('cuda:0')

googlenet = models.googlenet(pretrained=True)
model = nn.Sequential(*list(googlenet.children())[0:4])

if args.avgpool_bool == True:
    model = nn.Sequential(model, nn.AvgPool2d(2, stride=1))

model.to(device)
model.eval()


calc_corr(model, track_l, ref_l, tracks, refs, device, args.stride, args.rot, args.start, args.end, args.scorefile)


if args.cmc == True:
    cmc_score = compute_cmc(score_mat)

    f, ax = plt.subplots(1)
    plt.plot(cmc_score)
    plt.xlabel('references')
    plt.ylabel('cmc-score')
    ax.set_ylim(bottom=0)
    plt.grid(True)
    f.savefig('C:\\Users\\User\\virtualtest\\MCNCC\\cmc_score_diagram')