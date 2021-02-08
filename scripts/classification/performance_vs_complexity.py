# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data, get_tf_device,
                                           get_git_root, import_real_data)
from master_scripts.analysis_functions import anodedata_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, MaxPooling2D,
                                     Dropout, InputLayer)
import tensorflow as tf
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 10,
        'batch_size': 64,
    },
    'random_seed': 120,
    'data': {
        'images_sim': "images_full_pixelmod.npy",
        'labels_sim': "labels_full.npy",
    },
}

# ================== Import Data ==================

# import simulated data
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + config['data']['images_sim'])
images = images.reshape(images.shape[0], 16, 16, 1)
labels = np.load(DATA_PATH + config['data']['labels_sim'])

# import real data
config_real = {
    'DATA_PATH': "../../data/real/",
    'DATA_FILENAME': "anodedata_500k.txt",
    'MODEL_PATH': "../../models/",
    'RESULTS_PATH': "../../results/",
    'CLASSIFIER': "367e35da671b.h5",
    'ENERGY_MODEL': "2137bd6d101c.h5",
    'POSITIONS_MODEL': "337cafc233f7.h5",
}


events, images_real = import_real_data(
    config_real['DATA_PATH'] + config_real['DATA_FILENAME'])  # Images not normalized
images_real = images_real.reshape(images_real.shape[0], 16, 16, 1)
images_real = normalize_image_data(images_real)  # Normalize images

# log-scale the images if desireable
config['scaling'] = "minmax"
if "np.log" in config['scaling']:
    images = np.log1p(images)

# set tf random seed
tf.random.set_seed(config['random_seed'])
experiments = {}
with tf.device(get_tf_device(20)):
    models = {}
    # Logistic
    model = Sequential()
    model.add(InputLayer(input_shape=(256,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['logistic'] = model

    # Small Dense network
    model = Sequential()
    model.add(InputLayer(input_shape=(256,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['dense_10'] = model

    # Large dense network
    model = Sequential()
    model.add(InputLayer(input_shape=(256,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['dense_100_5'] = model

    # Small convolutional network
    model = Sequential()
    model.add(
        Conv2D(10, kernel_size=3, activation='relu', input_shape=(16, 16, 1),
        padding='same')
    )
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    models['cnn_shallow'] = model
    
    # Large convolutional network, deeper and wider
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(16, 16, 1),
        padding='same')
    )
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['cnn_deep'] = model

    # Run experiments
    for k, m in models.items():
        print(m.summary())
        experiment = Experiment(
            model=m,
            config=config,
            model_type="classification",
            experiment_name="full_training_classifier_" + k
        )
        if "logistic" in k or "dense" in k:
            experiment.run(
            images.reshape(images.shape[0], 256),
            labels,
        )
        else:
            experiment.run(
                images,
                labels,
            )
        print("Outputting model:", k)
        experiment.save()
        mpath = experiment.config['path_args']['models'] + experiment.id + ".h5"
        experiment.model.save(mpath)
        experiments[k] = experiment.id

        # Predict on experimental data and output results
        if "logistic" in k or "dense" in k:
            pred = experiment.model.predict(
                images_real.reshape(images_real.shape[0], 256)
            )
        else:
            pred = experiment.model.predict(
                images_real,
            )
        classification = (pred > 0.5).astype(int)
        tmp = anodedata_classification(events, classification)
        # Store the events as a json file
        out_filename = config_real['RESULTS_PATH'] \
            + "events_classified_" \
            + config_real["DATA_FILENAME"][:-4] \
            + "_C_" + k \
            + ".json"

        with open(out_filename, 'w') as fp:
            json.dump(tmp, fp, sort_keys=True, indent=4)

print("Performed experiments:")
for k, v in experiments.items():
    print(k, ":", v)
