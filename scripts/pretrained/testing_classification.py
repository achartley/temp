# Imports
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from analysis_functions.function import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, StratifiedKFold



# Load existing data. The feature_rep functions just use numpy storage.
# Image are allready reshaped to (16, 16, 3) here
images = load_feature_representation("images_noscale_200k.npy")
energies = load_feature_representation("energies_noscale_200k.npy")
positions = load_feature_representation("positions_noscale_200k.npy")
labels = load_feature_representation("labels_noscale_200k.npy")

images = normalize_image_data(images)
images = np.concatenate((images, images, images), axis=3)
n_classes = len(np.unique(labels))


def create_dense_model(input_shape):
    """ Create the same dense model that classifies on every pretrained net

    param input_shape: the shape of the input features

    return: the compiled model, ready for training etc.
    """
 
    model = Sequential()
    model.add(Dense(512, input_shape=pretrained_features.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Keys: model names, Values: depth to compare at.
pretrained_models = {
    "DenseNet121":None,
    "DenseNet169":None,
    "DenseNet201":None,
    "InceptionResNetV2":None,
    "InceptionV3":None,
    "MobileNet":None,
    "MobileNetV2":None,
    "NASNetLarge":None,
    "NASNetMobile":None,
    "ResNet50":None,
    "VGG16":None,
    "VGG19":None,
    "Xception":None,
    }


# Params for k-fold cross-validation
k_splits = 5
k_shuffle = True
epochs = 5
batch_size = 32
kfold_labels = labels.argmax(axis=-1) # StratifiedKFold doesn't take one-hot

# Store accuracy for each fold for all models
k_fold_results = {}
for key in pretrained_models.keys():
    k_fold_results[key] = []

# Store indices for double events that are never classified correctly by
# any model
never_correct = []

OUTPUT_PATH = "../../data/output/models/"

for net, depth in pretrained_models.items():
    print("Running for:", net)

    tmp_never_correct = []

    # variables for selecting best model for saving
    curr_max_acc = 0
    
    # Load features
    if depth is None:
        depth = "full"
    features_filename = net + "_d" + str(depth) + "_" + str(images.shape[0]) + "_new_" + ".npy"
    pretrained_features = load_feature_representation(features_filename)
    
    # Create KFold data generator
    skf = StratifiedKFold(n_splits=k_splits, shuffle=k_shuffle)
    
    # Run k-fold cv
    for train_index, test_index in skf.split(pretrained_features, kfold_labels):
        single_indices, double_indices, close_indices = event_indices(positions[test_index])
        
        # Build model
        model = create_dense_model(pretrained_features.shape[1:])

        # Setup callback for saving models
        fpath = OUTPUT_PATH + net + "-{val_accuracy:.2f}.hdf5"
        cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=fpath, 
                monitor='val_accuracy', 
                save_best_only=True
                )

        # Train model
        history = model.fit(
            pretrained_features[train_index], 
            labels[train_index], 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(pretrained_features[test_index], labels[test_index]),
            callbacks=[cb]
            )
        
        # Store the accuracy
        k_fold_results[net].append(history.history['val_accuracy'])

        # Get indices for wrongly classified double events
        tmp_pred = model.predict(pretrained_features[test_index])
        tmp_results = tmp_pred.argmax(axis=-1).reshape(tmp_pred.shape[0], 1)
        wrong_close = test_index[close_indices][np.where(tmp_results[close_indices] == 0)[0]]



        # Add wrongly classified double event indices to tmp storage
        tmp_never_correct = tmp_never_correct + wrong_close.tolist()

        # Delete models and temporary arrays to free memory
        del(model)
        del(cb)
        del(history)
        del(tmp_pred)
        del(tmp_results)


    # Set never_correct to the common elements of wrong_close events
    # and previous wrong_close events
    if len(never_correct) == 0:
        never_correct = set(tmp_never_correct)
    else:
        never_correct = never_correct.intersection(set(tmp_never_correct))

    # Delete pretrained features before loading new ones
    del(pretrained_features)
# Write results to file
with open("kfold_results.txt", "w") as resultfile:
    for key in k_fold_results.keys():
        accs = calc_kfold_accuracies(k_fold_results[key])
        resultfile.write("{}: acc_min = {:.2f} | acc_max = {:.2f} | acc_mean = {:.2f}\n".format(
            key, accs[0], accs[1], accs[2])
            )

# Write never_correct to file for later exploration
with open("never_correct_indices.txt", "w") as indexfile:
    for index in never_correct:
        indexfile.write(str(index) + "\n")

