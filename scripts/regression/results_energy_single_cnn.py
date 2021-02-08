# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data,
                                           normalize_position_data,
                                           event_indices,
                                           get_tf_device,
                                           get_git_root)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import tensorflow as tf
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
with open("results_experiment_config.json", 'r') as fp:
    config = json.load(fp)

# ================== Callbacks ====================


# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + config['data']['images'])
energies = np.load(DATA_PATH + config['data']['energies'])
positions = np.load(DATA_PATH + config['data']['positions'])

single_indices, double_indices, close_indices = event_indices(positions)
# log-scale the images if desireable
config['scaling'] = "minmax"
if "np.log" in config['scaling']:
    images = np.log1p(images)

# set tf random seed
tf.random.set_seed(config['random_seed'])
# ================== Import Data ==================
with tf.device(get_tf_device(20)):
    model = Sequential()
    model.add(
        Conv2D(8, kernel_size=3, activation='relu', input_shape=(16, 16, 1),
               padding='same')
    )
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    model.compile(
        loss='mse',
        optimizer='adam',
    )
    print(model.summary())

    # Run experiment
    experiment = Experiment(
        model=model,
        config=config,
        model_type="regression",
        experiment_name="generate_results_energies_single_cnn",
    )
    experiment.run_kfold(
        images[single_indices],
        energies[single_indices, 0],
    )
    experiment.save(save_model=True, save_indices=False)
    print("Finished experiment:", experiment.id)
