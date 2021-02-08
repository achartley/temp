# Imports
import os
import sys
sys.path.append("../../../master_scripts")
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data, get_tf_device,
                                           get_git_root,
                                           normalize_image_data_elementwise)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, MaxPooling2D,
                                     Dropout)
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 20,
        'batch_size': 64,
    },
    'random_seed': 120,
    'data': {
	'images': "images_training_18000000_202012080020.npy",
        'labels': "labels_training_18000000_202012080020.npy",
    },
}


# ================== Callbacks ====================


# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + config['data']['images'])
images = images.reshape(images.shape[0], 16, 16, 1)
labels = np.load(DATA_PATH + config['data']['labels'])

# log-scale the images if desireable
config['scaling'] = "minmax"
if "np.log" in config['scaling']:
    images = np.log1p(images)

# set tf random seed
tf.random.set_seed(config['random_seed'])
with tf.device(get_tf_device(20)):
    # Build model
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
        experiment_name="full_training_classifier"
    )
    experiment.run(
        images,
        labels,
    )
    experiment.save()
    mpath = experiment.config['path_args']['models'] + experiment.id + ".h5"
    model.save(mpath)
    print(experiment.id)
    lpath = experiment.config['path_args']['models'] + "models.log"
    log = open(lpath,"a")
    log.write(experiment.id + ":\n")
    log.write(os.path.basename(__file__)+"\n")
    log.write(repr(config) + "\n")
    log.close()
    
