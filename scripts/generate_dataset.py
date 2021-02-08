import warnings
import sys
sys.path.append('../../master_scripts')
from master_scripts.data_functions import generate_dataset_simulated
warnings.filterwarnings('ignore', category=FutureWarning)

DATA_PATH = "../data/simulated/"
fname = "CeBr20Mil_Mix.txt"

generate_dataset_simulated(
    DATA_PATH + fname,
    random_state=120,
    test_size=0.1
)
