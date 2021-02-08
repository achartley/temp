# Imports
from master_scripts.data_functions import (normalize_image_data,
                                           normalize_position_data,
                                           event_indices,
                                           get_git_root)
import autosklearn.regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
images = images.reshape(images.shape[0], 256)
positions = np.load(DATA_PATH + "positions_200k.npy")
labels = np.load(DATA_PATH + "labels_200k.npy")

single_indices, double_indices, close_indices = event_indices(positions)

train_idx, test_idx, non1, non2 = train_test_split(
    single_indices, single_indices, random_state=config['random_seed'])

# Convert positions to single-digit
positions = positions[:, 0]*16 + positions[:, 1]


automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=60,
    per_run_time_limit=15,
    tmp_folder='tmp',
    output_folder='output',
    ensemble_memory_limit=32768,
    ml_memory_limit=16384,
)
automl.fit(images[train_idx], positions[train_idx],
           dataset_name='testset-automl-singles')

print(automl.show_models())
predictions = automl.predict(images[test_idx])
print("R2 score:", r2_score(positions[test_idx], predictions))
