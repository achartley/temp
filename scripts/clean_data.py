from master_data_functions.functions import *
import numpy as np

filepath="../data/simulated/CeBr2Mil_Mix.txt"

separated_data = {}

# Temporary initialization of arrays-to-be
images = []
energies = []
positions = []
labels = []

# read line by line to alleviate memory strain when files are large
with open(filepath, "r") as infile:
    for line in infile:
        line = np.fromstring(line, sep=' ')
        image, energy, position = separate_simulated_data(line, False)
        label = label_simulated_data(line)

        images.append(image)
        energies.append(energy)
        positions.append(position)
        labels.append(label)

# Convert lists to numpy arrays and reshape them to remove the added axis 
images = np.array(images)
img_mean = np.mean(images)
img_bottom = np.amax(images) - np.amin(images)
images = images.reshape(images.shape[0], images.shape[2], images.shape[3], images.shape[4])
for i in range(images.shape[0]):
    try:
        normalized = (images[i] - img_mean)/img_bottom
    except RunTimeWarning:
        print("Runtime:", i)
        print(images[i], img_mean, img_bottom)

#energies = np.array(energies)
#positions = np.array(positions)
#labels = np.array(labels)

#energies = energies.reshape(energies.shape[0], energies.shape[2])
#positions = positions.reshape(positions.shape[0], positions.shape[2])
