import sys
sys.path.append('../../../master_scripts')
from master_scripts.data_functions import normalize_image_data
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, roc_auc_score)
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# TODO: Add in proper warning if one .json file is not added.

with open(sys.argv[1], 'r') as fp:
    config1 = json.load(fp)

configs = [config1]

for config in configs:

    images = np.load(config['DATA_PATH'] + config['IMAGE_FILE'])
    labels = np.load(config['DATA_PATH'] + config['LABEL_FILE'])
    positions = np.load(config['DATA_PATH'] + config['POSITION_FILE'])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    image_0 = ax[0].imshow(images[0].reshape(16, 16), origin='lower')
    ax[0].plot(positions[0, 0], positions[0, 1], 'rx')
    ax[0].plot(positions[0, 2], positions[0, 3], 'rx')
    ax[0].set_title("image 0, type: " + str(labels[0]))
    fig.colorbar(image_0, ax=ax[0])
    plt.show()

