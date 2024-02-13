import os
from os import path
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


def create_directories(folder):

    if not os.path.exists(path.join(folder, 'features')):

        try:
            os.mkdir(path.join(folder,'features'))
            os.mkdir(path.join(folder,'features/tracks'))
            os.mkdir(path.join(folder,'features/refs'))
            os.mkdir(path.join(folder,'features/refs_rot'))

        except OSError:
            print("Failed to create folders")
            print('Program terminated')
            sys.exit()
        else:
            print("Folders created")

    if not os.path.exists(path.join(folder, 'logs')):

        try:
            os.mkdir(path.join(folder,'logs'))
        except OSError:
            print("Failed to create folders")
            print('Program terminated')
            sys.exit()
        else:
            print("Folders created")




def create_model(avgpool_bool, avgp_stride):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    model = nn.Sequential(*list(googlenet.children())[0:4])

    if avgpool_bool:
        model = nn.Sequential(model, nn.AvgPool2d(2, stride=avgp_stride))

    model.to(device)
    model.eval()

    return device, model


def calculate_features(model, device, folder, track_l, tracks, ref_l, refs, skip_feat, start, end):

    trans = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not skip_feat:
        for x, t in enumerate(tqdm(np.sort(track_l))):
            template = Image.open(path.join(tracks, t))
            template_t = model(trans(template).unsqueeze(0).to(device))[0]
            template_t2 = template_t[:, 3:template_t.shape[1] - 3, 3:template_t.shape[2] - 3].cpu()
            np.save(folder + '/features/tracks/' + t.split(".")[0] + '.npy', template_t2.detach().numpy())

        for y, r in enumerate(tqdm(np.sort(ref_l))):
            image = Image.open(path.join(refs, r))
            image_t = model(trans(image).unsqueeze(0).to(device))
            image_t2 = image_t[:, :, 3:image_t.shape[2] - 3, 3:image_t.shape[3] - 3].cpu()
            np.save(folder + '/features/refs/'+ r.split(".")[0] + '.npy', image_t2.detach().numpy())

        for y, r in enumerate(tqdm(np.sort(ref_l))):
            img = Image.open(path.join(refs, r))
            for i in range(start, end):
                if i == start:
                    img2 = img.rotate(i, expand=1, fillcolor='white')
                    img2 = trans(img2).unsqueeze(0)
                    img_batch = torch.zeros(abs(start - end), 3, img2.shape[2], img2.shape[3])
                else:
                    img2 = img.rotate(i)
                    img2 = trans(img2).unsqueeze(0)
                    img_batch[i][:, :img2[0].shape[1], :img2[0].shape[2]] = img2[0]
            img_t_batch = model(img_batch.to(device)).cpu()
            np.save(folder + '/features/refs_rot/' + r.split(".")[0] + '.npy', img_t_batch.detach().numpy())

    else:
        print("Feature generation: skipped")




def compute_cmc(score_mat, label_file):

    test_labels = pd.read_csv(label_file, header=None)
    test_labels = test_labels[:][1].values.tolist()
    true_mat = np.zeros((len(test_labels), score_mat.shape[1]))


    for i in range(len(test_labels)):
        true_mat[i, test_labels[i] - 1] = 1

    cmc = np.zeros(score_mat.shape[1], dtype='float64')
    mx = np.zeros(score_mat.shape[0], dtype='float64')
    true_mat_est = np.zeros(score_mat.shape)
    est_loc = np.zeros(score_mat.shape[0])
    score_mat2 = score_mat

    for i in range(score_mat.shape[1]):

        for w in range(score_mat.shape[0]):
            mx[w] = max(score_mat2[w])


        for e in range(score_mat.shape[0]):
            true_mat_est[e] = np.equal(score_mat2[e], mx[e])

            est_loc[e] = list(true_mat_est[e]).index(1)
        if i == 0:
            with np.printoptions(threshold=np.inf):
                pass


        true_mat_est = true_mat_est * 1

        if i == 0:
            cmc[i] = np.tensordot(true_mat, true_mat_est, axes=2) / score_mat.shape[0]

        else:
            cmc[i] = (np.tensordot(true_mat, true_mat_est, axes=2) / score_mat.shape[0]) + cmc[i - 1]

        for g in range(score_mat.shape[0]):
            score_mat2[g][int(est_loc[g])] = -100000

    np.save('cmc_scores.npy', cmc)

    return cmc
