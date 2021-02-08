import sys
sys.path.append('../../master_scripts')
from master_scripts.noise_addition import *
import numpy as np
from master_scripts.data_functions import get_git_root
import json
import time

#config = {
#    'distfile': "ratiodist.txt",
#    'imagefile': ["images_training_18000000_202012080020.npy","images_test_2000000_202012080020.npy"]
#}

with open(sys.argv[1], 'r') as fp:
    config = json.load(fp)

DATA_PATH = get_git_root() + "data/simulated/"
DIST_PATH = get_git_root() + "data/real/"

for data in config['imagefile']:
    print(data)
    print(type(data))
    print("Noising " , data, " this may take some time.")
    start = time.time()
    images = np.load(DATA_PATH + data).copy()
    dist = gen_dist(DIST_PATH + config['distfile'])
    for i in range(0, len(images)):
        #print("START LOOP")
        #print(image[0][0:5])
        images[i] = images[i].reshape(16,16) * rnoise_gen(dist).reshape(16,16)
        #print('***************************')
        #print(image[0][0:5])
        #print("END LOOP")
    #np.save(images, 'n'+data)
    print(images[0])
    np.save('n'+data, images)
    end = time.time()
    print("Finished processing ", 'n'+data)
    print("Time elapsed:", end-start)
