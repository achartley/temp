# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (get_tf_device,
                                           get_git_root)
from master_scripts.models_pretrained import pretrained_model
from tensorflow.keras.layers import Dense
import json
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
with open("results_experiment_config.json", 'r') as fp:
    config = json.load(fp)

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

# For the pretrained vgg model we have to use data with 3 channels.
# This is solved by concatenating the images with themselves
images = np.concatenate((images, images, images), axis=-1)

# set tf random seed
tf.random.set_seed(config['random_seed'])
with tf.device(get_tf_device(20)):
    # Build model
    model = pretrained_model("VGG16", input_dim=(16, 16, 3))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
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
        experiment_name="full_training_pretrained_vgg16"
    )
    experiment.run_kfold(
        images,
        labels,
    )
    experiment.save(save_model=True, save_indices=False)
    print("Finished experiment:", experiment.id)
