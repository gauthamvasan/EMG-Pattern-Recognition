# Gesture recognition using SVM with Random Fourier Features
# This script loads every single mat file listed in the data directory.
# Each matrix has 6 categories - emg, repetition, restimulus, stimulus, glove, rerepetition
import os
import numpy as np
import scipy.io

dataDir = "/home/gautham/EMG-Pattern-Recognition/NINAPRO_Dataset/Database 1/Exercise 2/"
mats = []
for file in os.listdir( dataDir ) :
    mats.append( scipy.io.loadmat( dataDir+file ) )
print mats[0]
print (mats[0]['stimulus'].shape)
print (max(mats[0]['restimulus']))