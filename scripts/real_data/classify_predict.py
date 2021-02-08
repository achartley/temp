# Imports
import numpy as np
import tensorflow as tf
import sys
import json
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from models.models import *


# Load config
with open("config.json", "r") as infile:
    config = json.load(infile)

# Load data
data = import_real_data(config)

# Load models
classifier_path = config["MODEL_PATH"] + config["CLASSIFIER"]
energy_predictor_path = config["MODEL_PATH"] + config["SINGLE_ENERGY_MODEL"]
position_predictor_path = config["MODEL_PATH"] + config["SINGLE_POSITION_MODEL"]
classifier = tf.keras.models.load_model(classifier_path)
single_energy_model = tf.keras.models.load_model(single_energy_model_path)
single_position_model = tf.keras.models.load_model(single_position_model_path)

for event_id in data.keys():
    event_classification = classifier.predict(data[event_id]['image'])
    data[event_id]["event_class"] = event_classification.argmax(axis=-1)
    # If classified as single event, predict energy and position
    if data[event_id]["event_class"] == 0:
        event_energy = single_energy_model.predict(data[event_id]['image'])
        event_position = single_position_model.predict(data[event_id]['image'])
        data[event_id]["predicted_energy"] = event_energy
        data[event_id]["predicted_position"] = event_position

# Output results as json file
with open("results.json", "w") as fp:
    json.dump(data, fp, indent=2, sort_keys=True)


