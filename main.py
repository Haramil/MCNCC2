import os
from os import path
from matplotlib import pyplot as plt
import numpy as np
import argparse
import logging
import time

from functions.functions import create_directories, create_model, compute_cmc, calculate_features
from functions.normcorr import calc_corr

# Argparser
parser = argparse.ArgumentParser(description='take some individual folders from user')
parser.add_argument('-f', '--folder', type=str, help='define folder containing the dataset')
parser.add_argument('-t', '--tracks', type=str, default='tracks_cropped_Subset', help='define track folder')
parser.add_argument('-rf', '--refs', type=str, default='Subset', help='define reference folder')
parser.add_argument('-str', '--stride', type=int, default='1', help='stride for convolutions')
parser.add_argument('-avgp', '--avgpool_bool', default='False', action='store_true', help='activate average pooling for features')
parser.add_argument('-avgp_str', '--avgp_stride', type=int, default='1', help='stride for average_pooling')
parser.add_argument('-skf', '--skip_feat', default=False, action='store_true', help='skip feature generation')
parser.add_argument('-r', '--rot', default=False, action='store_true', help='add rotation')
parser.add_argument('-ris', '--start', type=int, default=-10, help='rotation interval start')
parser.add_argument('-rie', '--end', type=int, default=11, help='rotation interval end')
parser.add_argument('-sf', '--scorefile', type=str, default='scores.npy', help='scorefilename')
parser.add_argument('-cmc', '--cmc', default=False, action='store_true', help='calculate cmc')
parser.add_argument('-cmcf', '--cmc_file', type=str, default='cmc_file', help='cmc filename')
parser.add_argument('-lbltable', '--label_file', type=str, default='Subsetlabels.csv', help='name of the csv. file')

args = parser.parse_args()


# Paths
folder = args.folder
tracks = path.join(folder, args.tracks)
refs = path.join(folder, args.refs)

ref_l = [f for f in os.listdir(refs) if f.endswith('.png')]
track_l = [f for f in os.listdir(tracks) if f.endswith('.jpg')]


# Creating additional directories
create_directories(folder)


# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler(path.join(folder,'logs/mcncc.log'))
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)




if __name__ == "__main__":
    logger.info('folder: {}'.format(args.folder))
    logger.info('reference_folder: {}'.format(args.refs))
    logger.info('track_folder: {}'.format(args.tracks))
    logger.info('stride: {}'.format(args.stride))
    logger.info('skip_feat: {}'.format(args.skip_feat))
    logger.info('rotation: {}'.format(args.rot))
    logger.info('start: {}'.format(args.start))
    logger.info('end: {}'.format(args.end))
    logger.info('cmc: {}'.format(args.cmc))
    logger.info('scorefile: {}'.format(args.scorefile))
    logger.info('cmc_file: {}'.format(args.cmc_file))
    logger.info('label_file: {}'.format(args.label_file))
    logger.info('avg_bool: {}'.format(args.avgpool_bool))
    logger.info('avg_bool_stride: {}'.format(args.avgp_stride))

# Model
device, model = create_model(args.avgpool_bool, args.avgp_stride)

# Feature Calculation
calculate_features(model, device, folder, track_l, tracks, ref_l, refs, args.skip_feat, args.start, args.end)

# Feature paths
feature_tracks = folder + '/features/tracks'
feature_refs = folder + '/features/refs'
feature_refs_rot = folder + '/features/refs_rot'

feature_ref_l = [f for f in os.listdir(feature_refs) if f.endswith('.npy')]
feature_ref_rot_l = [f for f in os.listdir(feature_refs_rot) if f.endswith('.npy')]
feature_track_l = [f for f in os.listdir(feature_tracks) if f.endswith('.npy')]

# Calculation of the score matrix
calc_time = time.time()
score_mat = calc_corr(feature_track_l, feature_ref_l, feature_ref_rot_l, feature_tracks, feature_refs, feature_refs_rot, device, args.stride, args.rot, args.start, args.end, args.scorefile)
elapsed = time.time() - calc_time

logger.info("elapsed time: {}".format(elapsed))
logger.info("score_mat saved")

# CMC-score
if args.cmc:

    score_mat = np.load(args.scorefile)
    cmc_score = compute_cmc(score_mat, args.label_file)

    f, ax = plt.subplots(1)
    plt.plot(cmc_score)
    plt.xlabel('references')
    plt.ylabel('cmc-score')
    ax.set_ylim(bottom=0)
    plt.grid(True)
    f.savefig('cmc_figure')