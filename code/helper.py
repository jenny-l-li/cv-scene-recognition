#!/usr/bin/python

import os
import cv2
import numpy as np
import timeit, time
import random
import sys
from sklearn import neighbors, svm, cluster, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance


def getClusterCenters(descriptors, labels):
    vocabulary = []
    # map of label to 128D list of running total of descriptors (len dict_size x 128)
    label2des = {} 
    # map of label to number of descriptors for that label
    label2count = {}
    nrows = np.size(descriptors, 0)
    ncols = np.size(descriptors, 1)  # 128, 64, 32
    labels = labels.tolist()
    for label in labels:
        if not label in label2des:
            # initialize to all 0, 128 columns
            label2des[label] = [0] * ncols
            label2count[label] = 0

    for i, des in enumerate(descriptors):  # for each descriptor row 
        curr_label = labels[i]
        label2count[curr_label] = label2count[curr_label] + 1
        for j, d in enumerate(des):  # for each of the 128 col 
            label2des[curr_label][j] = label2des[curr_label][j] + d.item()

    for i in range(len(label2des)):
        label = label2des[i]
        for j in range(len(label)):
            label[j] = label[j] / label2count[i]
        vocabulary.append(label)
    return vocabulary
    