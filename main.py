import os
from os import path
from matplotlib import pyplot as plt
import argparse
import logging
#import torch
#import torchvision.models as models
#from torch import nn

from functions.model import create_model
from functions.cmc import compute_cmc
from functions.normcorr import calc_corr

folder = 'C:\\Users\\User\\virtualtest\\MCNCC2\\datasets\\FID-300'

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(message)s')

file_handler = logging.FileHandler('mcncc.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# Argparser
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

# Paths
tracks = path.join(folder, args.tracks)
refs = path.join(folder, args.refs)

ref_l = [f for f in os.listdir(refs) if f.endswith('.png')]
track_l = [f for f in os.listdir(tracks) if f.endswith('.jpg')]

if __name__ == "__main__":
    logger.info('refs: {}'.format(args.refs))
    logger.info('refs: {}'.format(args.tracks))
    logger.info('refs: {}'.format(args.stride))
    logger.info('refs: {}'.format(args.rot))
    logger.info('refs: {}'.format(args.start))
    logger.info('refs: {}'.format(args.end))
    logger.info('refs: {}'.format(args.cmc))
    logger.info('refs: {}'.format(args.scorefile))
    logger.info('refs: {}'.format(args.cmc_file))
    logger.info('refs: {}'.format(args.label_file))
    logger.info('refs: {}'.format(args.avgpool_bool))

# Model
device, model = create_model(args.avgpool_bool)

# Calculation of the score matrix
score_mat = calc_corr(model, track_l, ref_l, tracks, refs, device, args.stride, args.rot, args.start, args.end, args.scorefile)


# CMC-score
if args.cmc:
    cmc_score = compute_cmc(score_mat, folder, args.label_file)

    f, ax = plt.subplots(1)
    plt.plot(cmc_score)
    plt.xlabel('references')
    plt.ylabel('cmc-score')
    ax.set_ylim(bottom=0)
    plt.grid(True)
    f.savefig(cmc_score_diagram)