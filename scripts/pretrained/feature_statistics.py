# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from master_models.pretrained import pretrained_model

# Import data

# File import
# Sample filenames are:
# CeBr10kSingle_1.txt -> single events, 
# CeBr10kSingle_2.txt -> single events
# CeBr10k_1.txt -> mixed single and double events 
# CeBr10.txt -> small file of 10 samples
# CeBr2Mil_Mix.txt -> 2 million mixed samples of simulated events

# Flag import, since we can now import 200k events from .npy files
images = load_feature_representation("images_noscale_200k.npy")
energies = load_feature_representation("energies_noscale_200k.npy")
positions = load_feature_representation("positions_noscale_200k.npy")
labels = load_feature_representation("labels_noscale_200k.npy")

n_classes = labels.shape[1]
print("Number of classes: {}".format(n_classes))
print("Input Images shape: {}".format(images.shape))
print("Energies shape: {}".format(energies.shape))
print("Positions shape: {}".format(positions.shape))
print("Labels shape: {}".format(labels.shape))

# VGG16 expects 3 channels. Solving this by concatenating the image data 
# to itself, to form three identical channels

images = normalize_image_data(images)
images = np.concatenate((images, images, images), axis=3)
print("Reshaped Images data shape: {}".format(images.shape))

# Keys: model names, Values: depth to compare at.
pretrained_models = {
    #"DenseNet121":None, #8
    #"DenseNet169":None, #8
    #"DenseNet201":None, #8
    #"InceptionResNetV2":None, #8
    #"InceptionV3":None, #8
    #"MobileNet":None, #8
    #"MobileNetV2":None, #5
    #"NASNetLarge":None, #4
    #"NASNetMobile":None, #4
    "ResNet50":None, #8
    "VGG16":None,
    "VGG19":None,
    "Xception":None, #6
    }

# Define Kolmogorov-Smirnov test
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
# Check difference using Kolmogorov-Smirnov

def get_pval(i):
    ks = ks_2samp(pretrained_features[single_features][:,i], pretrained_features[double_features][:,i])
    return ks.pvalue


# Run test on all pretrained nets.
p_output = {}


for net, depth in pretrained_models.items():
    print("Running for:", net)
    
    # Load features
    if depth is None:
        depth = "full"
    features_filename = net + "_d" + str(depth) + "_" + str(labels.shape[0]) + "_new_" + ".npy"
    pretrained_features = load_feature_representation(features_filename)
    print(pretrained_features.shape)
    single_features = np.where(labels[:,0] == 1)[0]
    double_features = np.where(labels[:,1] == 1)[0]
    n = pretrained_features.shape[1]
    #p_values = Parallel(n_jobs=-1, verbose=2)(delayed(get_pval)(i) for i in range(n))
    p_values = []
    for i in range(n):
        print("{} / {}".format(i, n))
        p_values.append(get_pval(i))
    p_output[net] = p_values
    #plt.close()
    #plt.plot(range(len(p_values)), p_values, label=net)
    #plt.legend()
    #plt.savefig(net + "-p-vals.png")
    
    # Delete allocated arrays for saving memory in notebook
    del pretrained_features
    del single_features
    del double_features

# Get number of p-values below thresholds for each net to check
# if it's reasonable to reject the null-hypothesis (no difference in distributions)
ks_statistics = []
for key, val in p_output.items():
    pvals = np.array(p_output[key])
    n_features = len(pvals)
    n_below_1 = len(np.where(pvals < 0.01)[0])
    n_below_05 = len(np.where(pvals < 0.005)[0])
    n_below_01 = len(np.where(pvals < 0.001)[0])
    ks_statistics.append(
        [key, 
         n_features, 
         n_below_1/n_features,
         n_below_05/n_features,
         n_below_01/n_features,
        ]
    )
from tabulate import tabulate
# Output as latex table
headers = ["Network", "num_features", "ratio p < 0.01", "ratio p < 0.005", "ratio p < 0.001"]
print(tabulate(ks_statistics, headers, tablefmt="latex"))
