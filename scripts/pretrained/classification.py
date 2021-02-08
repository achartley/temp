# Imports
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
# Load existing data. The feature_rep functions just use np storage,
images = load_feature_representation("images_200k.npy")
energies = load_feature_representation("energies_200k.npy")
positions = load_feature_representation("positions_200k.npy")
labels = load_feature_representation("labels_200k.npy")

n_classes = labels.shape[1]

# VGG16 expects 3 channels. Solving this by concatenating the image data 
# to itself, to form three identical channels
images = np.concatenate((images, images, images), axis=3)

# Keys: model names, Values: depth to compare at.
pretrained_models = {
    "DenseNet121":None,
    "DenseNet169":None,
    "DenseNet201":None,
    "InceptionResNetV2":None,
    "InceptionV3":None,
    "MobileNet":None,
    "MobileNetV2":None,
    "NASNetLarge":None,
    "NASNetMobile":None,
    "ResNet50":None,
    "VGG16":None,
    "VGG19":None,
    "Xception":None,
    }

for net, depth in pretrained_models.items():
    print("Running for:", net)
    
    # Load features
    if depth is None:
        depth = "full"
    features_filename = net + "_d" + str(depth) + "_" + str(images.shape[0]) + ".npy"
    pretrained_features = load_feature_representation(features_filename)

    
