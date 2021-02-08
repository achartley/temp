import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from master_models.prediction import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend


# ================== Custom Functions ==================
# Define R2 score for metrics since it's not available by default
def r2_keras(y_true, y_pred):
    SS_res =  backend.sum(backend.square(y_true - y_pred)) 
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + backend.epsilon()) )

# Load model
MODEL_PATH = "../../data/output/models/"
FIGURE_PATH = "../../"
DATA_PATH = "../../data/simulated/"

# Load data
images = np.load(DATA_PATH + "images_1M.npy")
positions = np.load(DATA_PATH + "positions_1M.npy")
energies = np.load(DATA_PATH + "energies_1M.npy")
targets = np.concatenate((positions/16, energies), axis=1)
with tf.device('/GPU:2'):
    name = "cnn_pos_energy-r2-0.89.hdf5"
    loaded_model = tf.keras.models.load_model(MODEL_PATH+name, custom_objects={'r2_keras': r2_keras})
    
    
    y_pred = loaded_model.predict(images)
    y_resid = targets - y_pred
    
single, double, close = event_indices(positions)
rel_dist = relative_distance(positions)/16
rel_E = relative_energy(energies)

indices = np.random.choice(double, 10000, replace=False)

# Residuals in pos 1 vs residuals in energy 1
pos1_resid = np.sum(y_resid[:, :2], axis=1)
plt.scatter(y_resid[indices, 4], pos1_resid[indices], alpha=0.2)
plt.xlabel("Residual energy")
plt.ylabel("Residual sum(x1,y1)")
plt.savefig("resid_e1_pos1.pdf")
plt.clf()

# Residuals in pos 2 vs residuals in energy 2
pos2_resid = np.sum(y_resid[:, 2:4], axis=1)
plt.scatter(y_resid[indices, 5], pos2_resid[indices], alpha=0.2)
plt.xlabel("Residual energy")
plt.ylabel("Residual sum(x2,y2)")
plt.savefig("resid_e2_pos2.pdf")
plt.clf()

#
#plt.scatter(rel_E[indices], y_resid[indices], alpha=0.2)
#plt.xlabel("rel_E")
#plt.ylabel("sum_residuals")
#plt.savefig("pos_energy_residuals_rel_e.pdf")
#plt.clf()
#    
#plt.scatter(rel_dist[indices], y_resid[indices], alpha=0.2)
#plt.xlabel("rel_dist")
#plt.ylabel("sum_residuals")
#plt.savefig("pos_energy_residuals_rel_dist.pdf")
