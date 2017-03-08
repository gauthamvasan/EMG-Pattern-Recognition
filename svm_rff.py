# Gesture recognition using SVM with Random Fourier Features
# This script loads every single mat file listed in the data directory.
# Each matrix has 6 categories - emg, repetition, restimulus, stimulus, glove, rerepetition
import os
import numpy as np
import scipy.io
import tensorflow as tf

path = "/home/gautham/EMG-Pattern-Recognition/NINAPRO_Dataset/"
mats = []
sub_dirs = [x[0] for x in os.walk(path)]

# # Load the required file from every sub folder
# for sub_dir in sub_dirs:
#     sub_path = os.path.join(path,sub_dir)
#     for i in os.listdir(sub_path):
#         file_path = os.path.join(sub_path,i)
#         if os.path.isfile(file_path) and 'E1' in i:
#             mats.append(scipy.io.loadmat(file_path))

def segmentation(emg, samples = 300):
    N = samples # number of samples per segment
    S = int(np.floor(emg.shape[0]/N)) # number of segments
    length = 0
    segmented_emg = np.zeros((N,S))
    for s in range(S):
        for n in range(length,N+length):
            segmented_emg[n-length,s] = emg[n] # NxS matrix with a EMG signal divided in s segments, each one with n samples
        length = length + N
    length = 0
    return segmented_emg

a = tf.add(1, 2,)
b = tf.multiply(a, 3)
c = tf.add(4, 5,)
d = tf.multiply(c, 6,)
e = tf.multiply(4, 5,)
f = tf.div(c, 6,)
g = tf.add(b, d)
h = tf.multiply(g, f)

with tf.Session() as sess:
	writer = tf.summary.FileWriter("output", sess.graph)
	print(sess.run(h))
	writer.close()

# Examples to access the attributes of the data
# print (mats[0]['stimulus'].shape)
# print (max(mats[0]['restimulus']))