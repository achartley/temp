# Imports
import numpy as np
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Import Data ==================
#images = np.load(DATA_PATH + "images_noscale_200k.npy")
positions = np.load(DATA_PATH + "positions_noscale_200k.npy")
energies = np.load(DATA_PATH + "energies_noscale_200k.npy")
#images = normalize_image_data(images)
#images = np.load(DATA_PATH + "images_1M.npy")
#positions = np.load(DATA_PATH + "positions_1M.npy")
#energies = np.load(DATA_PATH + "energies_1M.npy")
#labels = np.load(DATA_PATH + "labels_noscale_200k.npy")

# ================== Prepare Data ==================
#images = images.reshape(images.shape[0], 256)
single, double, close = event_indices(positions)
rel_dist = relative_distance(positions)
rel_e = relative_energy(energies)
data = np.concatenate((positions, energies, rel_dist, rel_e), axis=1)


indices = np.random.choice(double, 10000, replace=False)

ticks = ["X1", "Y1", "X2", "Y2", "E1", "E2", "rel_dist", "rel_E"]
corr_matrix = np.corrcoef(data[double], rowvar=False)
plt.imshow(corr_matrix)
plt.xticks(np.arange(len(ticks)), ticks)
plt.yticks(np.arange(len(ticks)), ticks)
plt.colorbar()
plt.title("Correlation matrix for target positions and energies")
plt.savefig("corr_matrix.pdf")
plt.clf()

#plt.scatter(energies[indices,0], energies[indices,1], alpha=0.2)
#plt.savefig("scatter_e.pdf")
#plt.clf()

#plt.scatter(rel_dist[indices], rel_e[indices], alpha=0.2)
#plt.savefig("scatter_rel.pdf")
#plt.clf()


