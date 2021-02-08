# ============================================================================
# Modifies pixels in the simulated detector images in order to make it more
# similar to the real data.
# In anodedata_500k.txt pixel 3, 13 is set to 0 due to being unreliably noisy
# The same is the case for the bordering pixels y=0 and y=15
# ============================================================================
import numpy as np
from master_scripts.data_functions import get_git_root

repo_root = get_git_root()

images = np.load(repo_root + "data/simulated/images_full.npy")

images[:, 3, 13] = 0
images[:, 0, :] = 0
images[:, 15, :] = 0

np.save(repo_root + "data/simulated/images_full_pixelmod.npy", images)
