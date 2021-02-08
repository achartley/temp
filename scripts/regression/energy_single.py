# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data,
                                           event_indices,
                                           get_tf_device,
                                           get_git_root)
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 10,
        'batch_size': 32,
    },
    'random_seed': 120,
    'data': "full_pixelmod",
}

# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + f"images_{config['data']}.npy")
images = images.reshape(images.shape[0], 16, 16, 1)
positions = np.load(DATA_PATH + "positions_full.npy")
energies = np.load(DATA_PATH + "energies_full.npy")
labels = np.load(DATA_PATH + "labels_full.npy")

single_indices, double_indices, close_indices = event_indices(positions)

print("Loaded data")
# log-scale the images if desireable
config['scaling'] = "minmax"
if "np.log" in config['scaling']:
    images = np.log1p(images)
# set tf random seed
tf.random.set_seed(config['random_seed'])
search_name = "energy_regression_single_seeded"
with tf.device(get_tf_device(20)):
    padding = 'same'
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(16, 16, 1),
                     padding=padding))
    model.add(Conv2D(64, (3, 3), activation='relu', padding=padding))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding=padding))
    model.add(Conv2D(64, (3, 3), activation='relu', padding=padding))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
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
        experiment_name=search_name
    )
    experiment.run(
        normalize_image_data(images[single_indices]),
        energies[single_indices, 0],
    )
    experiment.save()
    mpath = experiment.config['path_args']['models'] + experiment.id + ".h5"
    model.save(mpath)
