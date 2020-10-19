import os
from os import path
import numpy as np
import pandas as pd

def compute_cmc(score_mat, folder, label_file):


    test_labels = pd.read_csv(path.join(folder, label_file), header=None)
    test_labels = test_labels[:][1].values.tolist()

    true_mat = np.zeros((len(test_labels), 38))

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
                print(est_loc)

        true_mat_est = true_mat_est * 1

        if i == 0:
            cmc[i] = np.tensordot(true_mat, true_mat_est, axes=2) / score_mat.shape[0]

        else:
            cmc[i] = (np.tensordot(true_mat, true_mat_est, axes=2) / score_mat.shape[0]) + cmc[i - 1]

        for g in range(score_mat.shape[0]):
            score_mat2[g][int(est_loc[g])] = -100000

    np.save("C:\\Users\\User\\virtualtest\\MCNCC\\" + args.cmc_file, cmc)

    return cmc