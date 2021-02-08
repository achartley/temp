# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data, get_tf_device,
                                           get_git_root)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 20,
        'batch_size': 32,
    },
    'random_seed': 120,
}

# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + "images_200k.npy")
images = images.reshape(images.shape[0], 16, 16, 1)
labels = np.load(DATA_PATH + "labels_200k.npy")

x_idx = np.arange(images.shape[0])
train_idx, val_idx, u1, u2 = train_test_split(
    x_idx, x_idx, random_state=config['random_seed']
)

# ================== Search params ================
batch_sizes = [32, 64, 128, 256]

# set tf random seed
tf.random.set_seed(config['random_seed'])
id_param = {}
search_name = "batch_size_deeper"
with tf.device(get_tf_device(20)):
    for b_size in batch_sizes:
        config['fit_args']['batch_size'] = b_size
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu', input_shape=(16, 16, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Run experiment
        experiment = Experiment(
            model=model,
            config=config,
            model_type="classification",
            experiment_name=search_name
        )
        experiment.run(
            normalize_image_data(images[train_idx]),
            labels[train_idx],
            normalize_image_data(images[val_idx]),
            labels[val_idx],
        )
        experiment.save()
        id_param[experiment.id] = {
            'batch_size': b_size,
        }
search_path = get_git_root() + "experiments/searches/"
with open(search_path + search_name + ".json", "w") as fp:
    json.dump(id_param, fp, indent=2)
