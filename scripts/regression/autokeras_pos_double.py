# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data,
                                           normalize_position_data,
                                           event_indices,
                                           get_tf_device,
                                           get_git_root)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import autokeras as ak
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
    'data': "200k",
}

# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + f"images_{config['data']}.npy")
images = images.reshape(images.shape[0], 16, 16, 1)
positions = np.load(DATA_PATH + "positions_200k.npy")
labels = np.load(DATA_PATH + "labels_200k.npy")

single_indices, double_indices, close_indices = event_indices(positions)
train_idx, val_idx, non1, non2 = train_test_split(
    double_indices, double_indices, random_state=config['random_seed'])
# log-scale the images if desireable
config['scaling'] = "minmax"
# set tf random seed
with tf.device(get_tf_device(20)):
	reg = ak.ImageRegressor(
	    overwrite=True,
	    max_trials=100,
	)
	# Feed the structured data regressor with training data.
	reg.fit(
	    normalize_image_data(images[train_idx]),
	    normalize_position_data(positions[train_idx]),
	    validation_data=(normalize_image_data(images[val_idx]), normalize_position_data(positions[val_idx])),
	    epochs=10,
	)
	predicted_y = reg.predict(normalize_image_data(images[val_idx]))
	print(reg.evaluate(normalize_image_data(images[val_idx]), normalize_position_data(positions[val_idx])))
