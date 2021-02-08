import sys
sys.path.append('../../../master_scripts')
from master_scripts.data_functions import normalize_image_data
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, roc_auc_score)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import json

# TODO: Add in proper warning if one .json file is not added.
# TODO: Make the number of files unlimited > 0


with open(sys.argv[1], 'r') as fp:
    config1 = json.load(fp)

with open(sys.argv[2], 'r') as fp:
    config2 = json.load(fp)


configs = [config1, config2]

for config in configs:

    images = np.load(config['DATA_PATH'] + config['IMAGE_FILE'])
    labels = np.load(config['DATA_PATH'] + config['LABEL_FILE'])
    #print("DEBUG: ", labels)
    print("DEBUG: ")
    print("CLASSIFIER: ", config['CLASSIFIER'])
    print("IMAGE_FILE: ", config['IMAGE_FILE'])

    if config['ML_METHOD'] == 'CNN':
        images = images.reshape(images.shape[0],16,16,1)
    else:
        images = images.reshape(images.shape[0], 256)

    model = tf.keras.models.load_model(config['MODEL_PATH'] + config['CLASSIFIER'])

    pred = model.predict(normalize_image_data(images))

    result = pred > 0.5

    accuracy = accuracy_score(labels, result)
    confmat = confusion_matrix(labels, result)
    f1 = f1_score(labels, result)
    mcc = matthews_corrcoef(labels, result)

    print("Model:", config['NAME'])
    print("Confusion matrix:\n", confmat)
    print("Accuracy:", accuracy)
    print("F1-score:", f1)
    print("MCC:", mcc)
