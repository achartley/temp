import sys
sys.path.append("../../../master_scripts")
from master_scripts.data_functions import (import_real_data,
                                           normalize_image_data)
import json
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data import

config = {
    'DATA_PATH': "../../data/real/",
    'DATA_FILENAME': "anodedata_500k.txt",
#    'DATA_PATH': "../../data/simulated/",
#    'DATA_FILENAME': "CeBr10k_1.txt",
    'MODEL_PATH': "../../models/",
    'RESULTS_PATH': "../../results/",
#    'CLASSIFIER': "367e35da671b.h5",
#    'ENERGY_MODEL': "2137bd6d101c.h5",
#    'POSITIONS_MODEL': "337cafc233f7.h5",
    'CLASSIFIER': "eb103c3cb3c4.h5",
    'ENERGY_MODEL': "eb103c3cb3c4.h5",
    'POSITIONS_MODEL': "eb103c3cb3c4.h5",
}

# Note this is not "real data" this is a test sanity check the models
#NPFILE = "../../data/simulated/CeBr10k_1.npy"

NPFILE = "anodedata_500k.npy"

print(config['DATA_PATH']+NPFILE)

events, images = import_real_data(
    config['DATA_PATH'] + config['DATA_FILENAME'])  # Images not normalized
#images = np.load(config['DATA_PATH'] + NPFILE)
images = images.reshape(images.shape[0], 16, 16, 1)
images = normalize_image_data(images)  # Normalize images

descriptors = list(
    set([event['event_descriptor'] for event in events.values()])
)

# Load models
model = tf.keras.models.load_model(
    config['MODEL_PATH'] + config['CLASSIFIER']
)

# Classify events
prediction = model.predict(images)
event_classification = (prediction > 0.5).astype(int)
for event_id in events.keys():
    if event_classification[events[event_id]['image_idx']] == 0:
        events[event_id]['event_class'] = "single"
    else:
        events[event_id]['event_class'] = "double"


desc_class = {
    'single': [],
    'double': [],
}
for event in events.values():
    desc_class[event['event_class']].append(event['event_descriptor'])

translate_descriptor = {
    1: "Implant",
    2: "Decay",
    3: "implant + Decay",
    4: "Light ion",
    5: "Implant + Light Ion",
    6: "Decay + Light Ion",
    7: "Implant + Decay + Light Ion",
    8: "Double (time)",
    9: "Implant + Double (time)",
    10: "Decay + Double (time)",
    11: "Implant + Decay + Double (time)",
    12: "Light ion + Double (time)",
    13: "Implant + Light Ion + Double (time)",
    14: "Decay + Light ion + Double (time)",
    15: "Implant + Decay + Light Ion + Double (time)",
    16: "Double (space)",
    17: "Implant + Double (space)",
    18: "Decay + Double (space)"
}
# Print a table-like structure for viewing
print("Classification results:")
print("|Event descriptor | Event type                   | singles | doubles |")
print("| :---           |  :---:                       | :---:   | :---:   |")
for d in descriptors:
    print("|{:^17d}|{:^30s}|{:^9d}|{:^9d}|".format(
        d,
        translate_descriptor[d],
        desc_class['single'].count(d),
        desc_class['double'].count(d)))

# ============================================================================
# SINGLE EVENTS - energy prediction
# ============================================================================
# Get single indices
single_indices = []
single_keys = []
for k in events.keys():
    if events[k]['event_class'] == 'single':
        single_indices.append(events[k]['image_idx'])
        single_keys.append(k)
single_indices = np.array(single_indices)

# Load model
model = tf.keras.models.load_model(
    config['MODEL_PATH'] + config['ENERGY_MODEL']
)

# Predict on single events
print("Predicting energies...")
prediction = model.predict(images[single_indices])
for i, k in enumerate(single_keys):
    events[k]['predicted_energy'] = prediction[i].tolist()
# ============================================================================
# SINGLE EVENTS - position prediction
# ============================================================================
# Load model
model = tf.keras.models.load_model(
    config['MODEL_PATH'] + config['POSITIONS_MODEL']
)

print("Predicting positions...")
prediction = model.predict(images[single_indices])
for i, k in enumerate(single_keys):
    events[k]['predicted_position'] = prediction[i].tolist()
# Store the events as a json file
out_filename = config['RESULTS_PATH'] \
    + "events_classified_" \
    + config["DATA_FILENAME"][:-4] \
    + "_C_" + config["CLASSIFIER"][:-3] \
    + "_E_" + config["ENERGY_MODEL"][:-3] \
    + "_P_" + config["POSITIONS_MODEL"][:-3] \
    + "_gaussian.json"

with open(out_filename, 'w') as fp:
    json.dump(events, fp, sort_keys=True, indent=4)
print("Output file to: ", out_filename)
