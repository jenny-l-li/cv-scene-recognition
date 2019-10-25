#!/usr/bin/python

from utils import *
import argparse

parser = argparse.ArgumentParser(description='CS188.2 - Fall 19 - Homework 1')
parser.add_argument("--tiny", "-t", type=bool, default=True, help='run Tiny Images')
parser.add_argument("--create-path", "-cp", type=bool, default=True, help='create the Results directory')
args = parser.parse_args()

# The argument is included as an idea for debugging, with a few examples in the main. Feel free to modify it or add arguments.
# You are also welcome to disregard this entirely

#############################################################################################################################
# This file contains the main. All the functions you need to write are included in utils. You also need to edit the main.
# The main just gets you started with the data and highlights the high level structure of the project.
# You are free to modify it as you wish - the modifications you are required to make have been marked but you are free to make
# others.
# What you cannot modify is the number of files you have to save or their names. All the save calls are done for you, you
# just need to specify the right data.
#############################################################################################################################


if __name__ == "__main__":
    
    if args.create_path:
        # To save accuracies, runtimes, voabularies, ...
        if not os.path.exists('Results'):
            os.mkdir('Results') 
        SAVEPATH = 'Results/'
    
    # Load data, the function is written for you in utils
    train_images, test_images, train_labels, test_labels = load_data()
    # print 'labels', train_labels
    
    # if args.tiny:
    #     tinyRes = tinyImages(train_images, test_images, train_labels, test_labels)
    
    #     # Split accuracies and runtimes for saving  
    #     for element in tinyRes[::2]:
    #         # Check that every second element is an accuracy in reasonable bounds
    #         assert (7 < element and element < 21)
    #     acc = np.asarray(tinyRes[::2])
    #     runtime = np.asarray(tinyRes[1::2])
    
    #     # Save results
    #     np.save(SAVEPATH + 'tiny_acc.npy', acc)
    #     np.save(SAVEPATH + 'tiny_time.npy', runtime)

    # Create vocabularies, and save them in the result directory
    vocabularies = []
    vocab_idx = [] # If you have doubts on which index is mapped to which vocabulary, this is referenced here
    # e.g vocab_idx[i] will tell you which algorithms/neighbors were used to compute vocabulary i
    # This isn't used in the rest of the code so you can feel free to ignore it
    buildDict_time = []
    bow_time = []

    # vocabs built: 
    # sift kmeans 20 50 
    # surf kmeans 20 50 
    # orb kmeans 20 50

    # #for feature in ['sift', 'surf', 'orb']:
    # for feature in ['sift']:
    #     for algo in ['kmeans']:
    #     # for algo in ['kmeans', 'hierarchical']:
    #     # dict_size [20, 50]
    #         for dict_size in [20, 50]:
    #             start = timeit.default_timer() 
    #             vocabulary = buildDict(train_images, dict_size, feature, algo)
    #             end = timeit.default_timer() 
    #             buildDict_time.append(round((end - start), 2))
    #             filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
    #             np.save(SAVEPATH + filename, np.asarray(vocabulary))
    #             vocabularies.append(vocabulary) # A list of vocabularies (which are 2D arrays)
    #             vocab_idx.append(filename.split('.')[0]) # Save the map from index to vocabulary
    
    print 'building...'
    for feature in ['surf']:
        for algo in ['kmeans']:
            for dict_size in [20, 50]:
                start = timeit.default_timer() 
                vocabulary = buildDict(train_images, dict_size, feature, algo)
                end = timeit.default_timer() 
                t = round((end - start), 2)
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                np.save(SAVEPATH + filename, np.asarray(vocabulary))
                filename2 = 'buildd_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                np.save(SAVEPATH + filename2, t)
                print 'almost done'
    print 'build done'

    # get existing vocabs
    for feature in ['sift', 'surf', 'orb']:
        for algo in ['kmeans']:
            for dict_size in [20, 50]:
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                vocabularies.append(np.load(SAVEPATH + filename))
                filename2 = 'buildd_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                if feature == 'orb' or feature == 'surf':
                    print np.load(SAVEPATH + filename2)


    # print 'buildd time', buildDict_time
    # Compute the Bow representation for the training and testing sets
    test_rep = [] # To store a set of BOW representations for the test images (given a vocabulary)
    train_rep = [] # To store a set of BOW representations for the train images (given a vocabulary)
    features = ['sift'] * 4 + ['surf'] * 4 + ['orb'] * 4 # Order in which features were used 
    # for vocabulary generation
    # 12 vocabularies

    # custom computebow 
    # 0 1 2 3 | 4 5 6 7 | 8 9 10 11 
    # k20 k50 h20 h50
    start = timeit.default_timer() 
    for image in train_images: # Compute the BOW representation of the training set
        rep = computeBow(image, vocabularies[2], 'surf') # Rep is a list of descriptors for a given image
        train_rep.append(rep)
    np.save(SAVEPATH + 'bow_train_' + str(4) + '.npy', np.asarray(train_rep)) # Save the representations for vocabulary i
    train_rep = [] # reset the list to save the following vocabulary
    for image in test_images: # Compute the BOW representation of the testing set
        rep = computeBow(image, vocabularies[2], 'surf')
        test_rep.append(rep)
    np.save(SAVEPATH + 'bow_test_' + str(4) + '.npy', np.asarray(test_rep)) # Save the representations for vocabulary i
    test_rep = [] # reset the list to save the following vocabulary
    end = timeit.default_timer() 
    t = round((end - start), 2)
    np.save(SAVEPATH + 'bowtime_' + str(4) + '.npy', t) 
    bow_time.append(round((end - start), 2))
    
    start = timeit.default_timer() 
    for image in train_images: # Compute the BOW representation of the training set
        rep = computeBow(image, vocabularies[3], 'surf') # Rep is a list of descriptors for a given image
        train_rep.append(rep)
    np.save(SAVEPATH + 'bow_train_' + str(5) + '.npy', np.asarray(train_rep)) # Save the representations for vocabulary i
    train_rep = [] # reset the list to save the following vocabulary
    for image in test_images: # Compute the BOW representation of the testing set
        rep = computeBow(image, vocabularies[3], 'surf')
        test_rep.append(rep)
    np.save(SAVEPATH + 'bow_test_' + str(5) + '.npy', np.asarray(test_rep)) # Save the representations for vocabulary i
    test_rep = [] # reset the list to save the following vocabulary
    end = timeit.default_timer() 
    t = round((end - start), 2)
    np.save(SAVEPATH + 'bowtime_' + str(5) + '.npy', t) 
    bow_time.append(round((end - start), 2))

    # for i, vocab in enumerate(vocabularies):
    #     if features[i] == 'sift' and i < 2:  # sift kmeans 20 50
    #         continue
    #     start = timeit.default_timer() 
    #     for image in train_images: # Compute the BOW representation of the training set
    #         rep = computeBow(image, vocab, features[i]) # Rep is a list of descriptors for a given image
    #         train_rep.append(rep)
    #     np.save(SAVEPATH + 'bow_train_' + str(i) + '.npy', np.asarray(train_rep)) # Save the representations for vocabulary i
    #     train_rep = [] # reset the list to save the following vocabulary
    #     for image in test_images: # Compute the BOW representation of the testing set
    #         rep = computeBow(image, vocab, features[i])
    #         test_rep.append(rep)
    #     np.save(SAVEPATH + 'bow_test_' + str(i) + '.npy', np.asarray(test_rep)) # Save the representations for vocabulary i
    #     test_rep = [] # reset the list to save the following vocabulary
    #     end = timeit.default_timer() 
    #     bow_time.append(round((end - start), 2))
        
    # print 'bow time', bow_time


    # Use BOW features to classify the images with a KNN classifier
    # A list to store the accuracies and one for runtimes
    knn_accuracies = []
    knn_runtimes = []

    for i, vocab in enumerate(vocabularies):
        if i == 2:
            i = 4
        elif i == 3:
            i = 5
        elif i == 4:
            i = 8
        elif i == 5:
            i = 9
        bow_train = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
        bow_test = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')
        # print 'train', np.shape(bow_train), 'test', np.shape(bow_test), 'trainlab', len(train_labels), 'testlab', len(test_labels)
        start = timeit.default_timer() 
        predict = KNN_classifier(bow_train, train_labels, bow_test, 9); 
        end = timeit.default_timer() 
        knn_accuracies.append(reportAccuracy(test_labels, predict))  
        knn_runtimes.append(round((end - start), 2))
    
    print 'acc', knn_accuracies
    print 'run', knn_runtimes
    np.save(SAVEPATH+'knn_accuracies.npy', np.asarray(knn_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'knn_runtimes.npy', np.asarray(knn_runtimes)) # Save the runtimes in the Results/ directory
    
    # Use BOW features to classify the images with 15 Linear SVM classifiers
    lin_accuracies = []
    lin_runtimes = []
    svm_lambda = 10

    for i, vocab in enumerate(vocabularies):
        if i == 2:
            i = 4
        elif i == 3:
            i = 5
        elif i == 4:
            i = 8
        elif i == 5:
            i = 9
        bow_train = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
        bow_test = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')
        start = timeit.default_timer()
        predict = SVM_classifier(bow_train, train_labels, bow_test, True, svm_lambda)
        end = timeit.default_timer() 
        lin_accuracies.append(reportAccuracy(test_labels, predict))  
        lin_runtimes.append(round((end - start), 2))

    print 'accl', lin_accuracies
    print 'runl', lin_runtimes
    np.save(SAVEPATH+'lin_accuracies.npy', np.asarray(lin_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'lin_runtimes.npy', np.asarray(lin_runtimes)) # Save the runtimes in the Results/ directory
    
    # Use BOW features to classify the images with 15 Kernel SVM classifiers
    rbf_accuracies = []
    rbf_runtimes = []
    
    for i, vocab in enumerate(vocabularies):
        if i == 2:
            i = 4
        elif i == 3:
            i = 5 
        elif i == 4:
            i = 8
        elif i == 5:
            i = 9       
        bow_train = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
        bow_test = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')
        start = timeit.default_timer()
        predict = SVM_classifier(bow_train, train_labels, bow_test, False, svm_lambda)
        end = timeit.default_timer() 
        rbf_accuracies.append(reportAccuracy(test_labels, predict))  
        rbf_runtimes.append(round((end - start), 2))

    print 'accr', rbf_accuracies
    print 'runr', rbf_runtimes    
    np.save(SAVEPATH +'rbf_accuracies.npy', np.asarray(rbf_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH +'rbf_runtimes.npy', np.asarray(rbf_runtimes)) # Save the runtimes in the Results/ directory
    
    