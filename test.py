#!/usr/bin/python

import numpy as np
import cv2
from sklearn import datasets
from sklearn import svm
from sklearn import cluster
from matplotlib import pyplot as plt

## OpenCV

# Opening image
image = cv2.imread('img.jpg', cv2.IMREAD_COLOR)
print image.shape

# Saving image
print cv2.imwrite('img_copy.jpg', image)
image_copy = cv2.imread('img_copy.jpg', cv2.IMREAD_COLOR)
print np.all(image_copy == image)

# Cropping image
print image[0:100].shape  # row

# Resize
image_resize = cv2.resize(image, (80, 100))
print image_resize.shape


## Sklean

# SVM 
digits = datasets.load_digits()
clf = svm.SVC(kernel='linear')
new_labels = (digits.target[:-10] == 8).astype(np.int32)

# train SVM to fit training set
print clf.fit(digits.data[:-10], new_labels)

# predict, classify whether or not 8 
print clf.predict(digits.data[-10:])


# K-means clustering

blobs_X, blobs_y = datasets.make_blobs(centers=5, random_state=0)
kmeans = cluster.KMeans(5)

# fit kmeans to dataset (find cluster centers and assigns of points)
print kmeans.fit(blobs_X)

#predit the cluster the blobs belong to
cluster_labels = kmeans.predict(blobs_X)

print plt.scatter(blobs_X[:, 0], blobs_X[:, 1], c=cluster_labels)


