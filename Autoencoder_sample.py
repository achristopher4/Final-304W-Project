#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras import models
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from os import listdir
from os.path import isfile, join
from sklearn import metrics, mixture
import imageio.v2 as imageio


# Disables the GPU for tensorflow. If you don't have a valid GPU then this code does nothing.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Loads a sample of 100 data points from the negative and positive folders, adjust as needed.
train_neg_des = './STEEL_TRAINING/POSITIVE'
train_pos_des = './STEEL_TRAINING/NEGATIVE'

neg_data = []
for extension in os.listdir(train_neg_des)[0:100]:
    neg_data += [imageio.imread(train_neg_des + "/" + extension)]
    
pos_data = []
for extension in os.listdir(train_pos_des)[0:100]:
    pos_data += [imageio.imread(train_pos_des + "/" + extension)]

neg_data = np.asarray(neg_data)
print(neg_data.shape)
pos_data = np.asarray(pos_data)
print(pos_data.shape)


# Loads the autoencoder model.
autoencoder = tf.keras.models.load_model('./autoencoder_model')

# Predicts new data using the model.
pos_reconstructed = autoencoder.predict(pos_data)


# Defines a function to display images, not needed for any actual functionality. 
def show_data(X, n=10, title="", cmap='viridis'):
    plt.figure(figsize=(15, 5), dpi=80)
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(tf.keras.utils.array_to_img(X[i]), cmap=plt.get_cmap(cmap))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)


# Visually compare images.
# This works when using Jupyter, but I haven't tested the results when running as a .py file in Visual Studio or PyCharm.
show_data(pos_data, title="Original Image", n=5)
show_data(pos_reconstructed, title="Reconstructed Image", n=5)
pos_recon_error=(np.abs(pos_data-pos_reconstructed))
show_data(pos_recon_error, title="Reconstruction Error", n=5)





