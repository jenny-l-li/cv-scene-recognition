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


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    # ignore any file name with leading '.'
    train_classes = sorted([dirname for dirname in os.listdir(train_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
    
    # training size: 1500
    # testing size: 2985
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(train_features, train_labels)
    predicted_categories = knn.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    fifteen_svms = []
    fifteen_set = set(train_labels)
    fifteen_labels = list(fifteen_set)
    fifteen_train = []  # new training labels: 15 x N
    predicted_categories = []
    x = 0
    if is_linear:
        clf = svm.LinearSVC(C=svm_lambda, multi_class='ovr')
        clf.fit(train_features, train_labels)
        predicted_categories = clf.predict(test_features)
    # else:
    #     clf = svm.SVC(C=svm_lambda, kernel='rbf', decision_function_shape='ovr', gamma='scale')  # gamma='scale'
    #     clf.fit(train_features, train_labels)
    #     predicted_categories = clf.predict(test_features)
    #     print 'yit'
    else:
        # create list of 15 training label lists fifteen_train:
        # [[label0, none, none, label0...]
        # [none, label1, none, none...] etc.
        for label in fifteen_labels:
            temp = []
            for l in train_labels:
                if label == l:
                    temp.append(label)
                else:
                    temp.append(-1)
            fifteen_train.append(temp)

        # train each of the 15 svms
        for i in range(15):
            clf = svm.SVC(C=svm_lambda, kernel='rbf', decision_function_shape='ovr', gamma='scale') 
            clf.fit(train_features, fifteen_train[i])
            fifteen_svms.append(clf)

        # classify each test image
        for test in test_features:
            max_conf = -1
            winner = None  # winning svm label with max confidence

            min_conf_none = sys.maxsize
            winner_none = None  # none of the labels win so pick the none with least conf

            # iterate through each of the 15 clfs
            for i, clf in enumerate(fifteen_svms):
                # print 'test', test
                pred_label = clf.predict(test.reshape(1, -1))
                conf = abs(clf.decision_function(test.reshape(1, -1)))
                # print 'pred', pred_label, 'i', i, 'conf', abs(conf), 'min_conf_none', min_conf_none
                if pred_label == -1 and winner is None:
                    if conf < min_conf_none:
                        min_conf_none = conf
                        winner_none = i
                elif pred_label == -1 and winner is not None:
                    continue
                else:
                    if conf > max_conf:
                        max_conf = conf
                        winner = pred_label

            if winner is not None:
                predicted_categories.append(winner)  # label with max confidence
                # print 'winner', winner
            else:
                predicted_categories.append(winner_none)
                # print 'winner none', winner_none

    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    output_image = cv2.resize(input_image, (target_size, target_size))
    return cv2.normalize(output_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    size = len(true_labels)
    # print size, len(predicted_labels)
    n_correct = 0
    for i in range(size):
        if true_labels[i] == predicted_labels[i]:
            n_correct +=1
    accuracy = round((n_correct / float(size)) * 100, 2)
    return accuracy

def getClusterCenters(descriptors, labels):
    vocabulary = []
    # map of label to 128D list of running total of descriptors (len dict_size x 128)
    label2des = {} 
    # map of label to number of descriptors for that label
    label2count = {}
    nrows = np.size(descriptors, 0)
    ncols = np.size(descriptors, 1)  # 128, 64, 32
    labels = labels.tolist()
    # print type(labels)
    for label in labels:
        if not label in label2des:
            # initialize to all 0, 128 columns
            label2des[label] = [0] * ncols
            label2count[label] = 0
    # print('map size {} des size {}'.format(len(label2des), nrows))
    # print('check descriptor and label len are same {} {}'.format(len(labels), nrows))

    # print 'ye', type(descriptors)
    for i, des in enumerate(descriptors):  # for each descriptor row 
        curr_label = labels[i]
        label2count[curr_label] = label2count[curr_label] + 1
        for j, d in enumerate(des):  # for each of the 128 col 
            label2des[curr_label][j] = label2des[curr_label][j] + d.item()

    #checkCount = 0
    for i in range(len(label2des)):
        label = label2des[i]
        #checkCount = checkCount + label2count[i]
        for j in range(len(label)):
            label[j] = label[j] / label2count[i]
        vocabulary.append(label)
    # print 'checkCount', checkCount
    return vocabulary

def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.

    vocabulary = []
    descriptors = [] # list of descriptors, changed to np array later
    limit_features = False
    if clustering_type == 'hierarchical':
        limit_features = True
    N_FEATURES = 25

    if feature_type == 'sift':
        if limit_features:
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=N_FEATURES)
        else: 
            sift = cv2.xfeatures2d.SIFT_create()
            print 'limit'

        # find keypoints and descriptors with SIFT for each image
        for img in train_images:
            # detect keypoints and compute descriptors
            # kp: list of keypoints
            # des: 2D numpy array of N_FEATURES x 128 (each 128D)
            kp, des = sift.detectAndCompute(img, None)
            for d in des:
                descriptors.append(d) 

        descriptors = np.asarray(descriptors)
        descriptors = cv2.normalize(descriptors, None)  # ???
        
        # KMeans
        if clustering_type == 'kmeans':
            # build the vocabulary: set of image features (words)
            kmeans = cluster.KMeans(n_clusters=dict_size).fit(descriptors)
            print('Shape of descriptors is {}'.format(np.shape(descriptors)))
            vocabulary = kmeans.cluster_centers_
            # print getClusterCenters(descriptors, kmeans.labels_)

        # Hierarchial Agglomerative 
        elif clustering_type == 'hierarchical':
            hier = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(descriptors)
            vocabulary = getClusterCenters(descriptors, hier.labels_)

    elif feature_type == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        for img in train_images:
            kp, des = surf.detectAndCompute(img, None)
            if limit_features:
                des_sample = random.sample(des, N_FEATURES)
            else:
                des_sample = des
            for d in des_sample:
                descriptors.append(d) 
            # print('sf Shape of des is {}'.format(np.shape(des_sample)))
        if clustering_type == 'kmeans':
            kmeans = cluster.KMeans(n_clusters=dict_size).fit(descriptors)
            print('Shape of descriptors is {}'.format(np.shape(descriptors)))
            vocabulary = kmeans.cluster_centers_
        elif clustering_type == 'hierarchical':
            hier = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(descriptors)
            vocabulary = getClusterCenters(descriptors, hier.labels_)                
    
    elif feature_type == 'orb':
        if limit_features:
            orb = cv2.ORB_create(nfeatures=N_FEATURES) # TODO: WTA_K=NORM_HAMMING
        else:
            orb = cv2.ORB_create()
        for img in train_images:
            kp, des = orb.detectAndCompute(img, None)
            if des is None:
                continue
            for d in des:
                descriptors.append(d) 
            print('o Shape of des is {}'.format(np.shape(des)))
        descriptors = np.asarray(descriptors)
        descriptors = cv2.normalize(descriptors, None)
        if clustering_type == 'kmeans':
            kmeans = cluster.KMeans(n_clusters=dict_size).fit(descriptors)
            vocabulary = kmeans.cluster_centers_
        elif clustering_type == 'hierarchical':
            hier = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(descriptors)
            vocabulary = getClusterCenters(descriptors, hier.labels_)

    print vocabulary
    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    # dimension: dict_size (# of clusters, bins)
    # value: how many times a descriptor was assigned to that cluster/bin

    # descriptor_dimension = {'sift': 128, 'surf': 64, 'orb': 32}
    # ncols = descriptor_dimension[feature_type]  

    dict_size = len(vocabulary)  # dict_size x 128
    # N_FEATURES = 25

    # extract features from image
    features = []
    if feature_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()  
        kp, descriptors = sift.detectAndCompute(image, None)
        for des in descriptors:
            features.append(des)  #128
        # print('BOW Shape of descriptors is {}'.format(np.shape(descriptors)))  # 25 x 128
    elif feature_type == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        kp, des = surf.detectAndCompute(image, None)
        for d in des:
            features.append(d)  #64
    else:  # 'orb'
        orb = cv2.ORB_create()  # TODO: WTA_K=NORM_HAMMING
        kp, des = orb.detectAndCompute(image, None)
        if des is None:
            features.append([0] * 32)
        else:
            for d in des:
                features.append(d)  #32

    bow = [0] * dict_size  # create bins
    TOTAL_FEATURES = len(features)
    # print 'tot features', TOTAL_FEATURES

    # find out what bin each feature goes in
    # cdist: compute Euclidean distance to each centroid of the vocabulary to find the bin
    for feature in features:  # feature: 1 x 128
        feature = np.reshape(feature, (1, -1))  # feature (1D) = [ feature ] (2D)
        res = distance.cdist(vocabulary, feature, 'euclidean')  # dict_size x 1
        # print 'cdist', res
        # bin index: min of distances 
        bin_index = np.where(res == np.amin(res))[0][0]
        # print 'bin_index', bin_index
        bow[bin_index] = bow[bin_index] + 1
    # print 'bow before norm', bow

    # normalize: each bin = values in each bin / total # of values
    for i, b1n in enumerate(bow):
        bow[i] = bow[i] / float(TOTAL_FEATURES)

    # print 'bow', bow
    return bow

def tinyImages_helper(train_features, test_features, train_labels, test_labels, train, test, scale, classResult):
    start = timeit.default_timer() 
    for img in train_features:
        train[scale].append((imresize(img, scale).flatten()).tolist())
    for img in test_features:
        test[scale].append((imresize(img, scale).flatten()).tolist())
    end = timeit.default_timer() 
    resize_time = (end - start)       

    start_1 = timeit.default_timer() 
    predict_1 = KNN_classifier(train[scale], train_labels, test[scale], 1);
    end_1 = timeit.default_timer() 
    classResult.append(reportAccuracy(test_labels, predict_1))
    classResult.append(round(resize_time + (end_1 - start_1), 2))

    start_3 = timeit.default_timer() 
    predict_3 = KNN_classifier(train[scale], train_labels, test[scale], 3);
    end_3 = timeit.default_timer() 
    classResult.append(reportAccuracy(test_labels, predict_3))
    classResult.append(round(resize_time + (end_3 - start_3), 2))

    start_6 = timeit.default_timer() 
    predict_6 = KNN_classifier(train[scale], train_labels, test[scale], 6);
    end_6 = timeit.default_timer() 
    classResult.append(reportAccuracy(test_labels, predict_6))
    classResult.append(round(resize_time + (end_6 - start_6), 2))
    return classResult


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds

    # train[#] test[#]: 2D list in which each row is a different image (post-resize)
    train = {}
    train[8] = []
    train[16] = []
    train[32] = []
    test = {}
    test[8] = []
    test[16] = []
    test[32] = []
    classResult = []

    classResult = tinyImages_helper(
            train_features, test_features, train_labels, test_labels, train, test, 8, classResult)
    classResult = tinyImages_helper(
            train_features, test_features, train_labels, test_labels, train, test, 16, classResult)
    classResult = tinyImages_helper(
            train_features, test_features, train_labels, test_labels, train, test, 32, classResult)
    print classResult
    return classResult
    