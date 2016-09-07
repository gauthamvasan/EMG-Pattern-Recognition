# Gesture recognition using SVM with Random Fourier Features
import os
import numpy as np
import scipy.io

dataDir = "/home/gautham/EMG-Pattern-Recognition/NINAPRO_Dataset/Database 1/Exercise 2/"
mats = []
for file in os.listdir( dataDir ) :
    mats.append( scipy.io.loadmat( dataDir+file ) )
print(len(mats))
print (mats[0]['stimulus'].shape)
print (mats[0]['stimulus'][1000])