import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay,
                             recall_score,
                             RocCurveDisplay)
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_LOC = "HCC_pkl_data"
LABELS_LOC = "HCC_description.xlsx"
TRAIN_AND_VAL_COMPRESSED_LOC = "HCC_compressed_data"
CV_RESULTS_LOC = "HCC_cv_results.xlsx"
MODEL_LOC = "HCC_net.joblib"

IMAGE_SIZE = (128, 128)
NUM_EPOCHS = 5
NUM_CHANNELS = 46**2 # must be a perfect square
BATCH_SIZE = 1
NUM_FOLDS = 5

COMPRESSED_FILES_CREATED = os.path.isdir(TRAIN_AND_VAL_COMPRESSED_LOC)

def process_raw_data(all_patient_data):
    """
    Takes in a list of numpy arrays, each representing a patient's data.
    
    Returns a list of numpy arrays, each representing a patient's data, with the following modifications:
    - All arrays have the same number of channels
    - All arrays have gaussian noise added to them
    - All arrays are normalized
    """
    
    # pad each batch with random image from same patient
    for i in tqdm(range(len(all_patient_data))):
        start_channels = all_patient_data[i].shape[-1]
        if start_channels < NUM_CHANNELS:
            batch_with_padding = np.zeros((*all_patient_data[i].shape[:-1], NUM_CHANNELS))
            batch_with_padding[..., :start_channels] = all_patient_data[i]
            
            for j in range(NUM_CHANNELS - start_channels):
                random_image_index = random.randint(0, start_channels-1)
                batch_with_padding[..., start_channels+j:start_channels+j+1] = all_patient_data[i][..., random_image_index:random_image_index+1]
            
            all_patient_data[i] = batch_with_padding
    
    # add gaussian noise to each image and normalize
    for i in tqdm(range(len(all_patient_data))):
        for j in range(all_patient_data[i].shape[-1]):
            all_patient_data[i][..., j:j+1] += np.random.normal(0, 0.01, all_patient_data[i][..., j:j+1].shape)
            all_patient_data[i][..., j:j+1] = (all_patient_data[i][..., j:j+1] - np.mean(all_patient_data[i][..., j:j+1])) / np.std(all_patient_data[i][..., j:j+1])
    
    return all_patient_data

def collect_data_and_kfolds(data_loc=DATA_LOC, labels_loc=LABELS_LOC, image_size=IMAGE_SIZE, num_folds=NUM_FOLDS, dest=TRAIN_AND_VAL_COMPRESSED_LOC):
    """
    Creates a list of numpy arrays, each representing a patient's data, and a list of labels for each patient.
    
    Each patient's data is a numpy array of shape (image_size[0], image_size[1], num_channels), where num_channels is the number of images in the patient's data.
    Each patient's label is a float, where 0 represents benign and 1 represents malignant.
    The data and labels are shuffled in the same order.
    The data is normalized and gaussian noise is added to each image.
    The data is split into num_folds folds, and the indexes of each fold are saved to a file.
    The data and labels are saved to files.
    """

    # get list of patients with data and sort by name
    patients_with_data, patient_paths = zip(*sorted([(patient.name, patient.path) for patient in os.scandir(data_loc) if patient.is_dir()]))

    # create labels dataframe, remove duplicates and patients without data
    labels_dataframe = pd.read_excel(labels_loc, usecols=["patient_code", "GT_benign_malignant", "fold"])
    labels_dataframe.drop_duplicates(subset=["patient_code"], inplace=True, ignore_index=True)
    labels_dataframe["patient_labels"] = (labels_dataframe["GT_benign_malignant"] == "malignant").astype(np.float32)
    labels_dataframe.drop("GT_benign_malignant", axis=1, inplace=True)
    labels_dataframe = labels_dataframe.loc[labels_dataframe["patient_code"].isin(patients_with_data)]

    # cannot allocate kfold_indexes in advance because each fold has a different number of patients
    kfold_indexes = [([], []) for _ in range(num_folds)]

    # allocate memory in advance for speed
    all_patient_data = np.empty(len(patient_paths), dtype=object)
    all_patient_labels = np.empty(len(patient_paths), dtype=object)

    for i, patient_path in enumerate(tqdm(patient_paths)):
        patient_fold_no = labels_dataframe["fold"].iloc[i] - 1
        for fold_no in range(num_folds):
            kfold_indexes[fold_no][int(fold_no != patient_fold_no)].append(i) # add index to training set if fold_no == patient_fold_no, else add to validation set
        
        patient_pkl_data = np.roll(sorted([data.path for data in os.scandir(patient_path)]), 1) # sort by name and roll to put washin first
        patient_np_data = np.concatenate([
                                tf.image.resize(
                                    np.swapaxes(
                                        np.load(filename, allow_pickle=True),2,0), image_size)
                                            for filename in patient_pkl_data], axis=-1) # load, resize, and concatenate all images along channels
        patient_label = labels_dataframe["patient_labels"].iloc[i]
        
        all_patient_data[i] = patient_np_data
        all_patient_labels[i] = patient_label
    
    [random.shuffle(index_list) for index_group in kfold_indexes for index_list in index_group] # shuffle indexes within each fold
    all_patient_data = process_raw_data(all_patient_data)

    all_patient_data = np.array(all_patient_data.tolist(), dtype=np.float32)
    all_patient_labels = np.array(all_patient_labels.tolist(), dtype=np.float32)
    kfold_indexes = np.array(kfold_indexes, dtype=object)

    Path(dest).mkdir(parents=True, exist_ok=True)
    with open(f"{dest}/train_and_validation_splits.npy", "wb") as splits_file:
        np.save(splits_file, kfold_indexes)
    
    with open(f"{dest}/train_and_validation_data.npy", "wb") as data_file:
        np.save(data_file, all_patient_data)
    
    with open(f"{dest}/train_and_validation_labels.npy", "wb") as labels_file:
        np.save(labels_file, all_patient_labels)

def define_model(num_channels, image_shape=IMAGE_SIZE):
    """
    Defines a model to be used for training.
    
    Returns a tf.keras.Sequential model.
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.Input((*image_shape, num_channels)))
    
    # reshape to (num_channels, image_size[0], image_size[1])
    model.add(tf.keras.layers.Permute((3, 1, 2)))
    
    # add dimension at the end -> num_channels, image_size[0], image_size[1], 1)
    model.add(tf.keras.layers.Reshape((num_channels, *image_shape, 1)))

    # grayscale to rgb -> (num_channels, image_size[0], image_size[1], 3)
    model.add(tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1)))

    # split num_channels across 2nd and 3rd dimensions -> (image_size[0]*sqrt(num_channels), image_size[1]*sqrt(num_channels), 3)
    model.add(tf.keras.layers.Reshape((image_shape[0]*int(np.sqrt(num_channels)), image_shape[1]*int(np.sqrt(num_channels)), 3)))

    pretrained_model = tf.keras.applications.MobileNetV3Small(include_top=False,
                                                              input_shape=(image_shape[0]*int(np.sqrt(num_channels)), image_shape[1]*int(np.sqrt(num_channels)), 3),
                                                              weights="imagenet")
    pretrained_model.trainable = False
    model.add(pretrained_model)
    model.add(tf.keras.layers.GroupNormalization())
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model

def cross_validate(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, num_folds=NUM_FOLDS, data_loc=TRAIN_AND_VAL_COMPRESSED_LOC, results_dest=CV_RESULTS_LOC, model_dest=MODEL_LOC):
    """
    Performs cross validation on the data.
    
    Prints the following metrics for each fold:
    - Sensitivity
    - Specificity
    - Confusion matrix
    - ROC curve
    """

    with open(f"{data_loc}/train_and_validation_splits.npy", "rb") as splits_file:
        cv_kfold_splits = np.load(splits_file, allow_pickle=True)

    with open(f"{data_loc}/train_and_validation_data.npy", "rb") as data_file:
        all_patient_data = np.load(data_file, allow_pickle=True)
    
    with open(f"{data_loc}/train_and_validation_labels.npy", "rb") as labels_file:
        all_patient_labels = np.load(labels_file, allow_pickle=True)

    for fold_no in range(num_folds):
        model = define_model(NUM_CHANNELS) # all_patient_data[0].shape[-1])

        train_data_and_labels = (all_patient_data[cv_kfold_splits[fold_no][0]].tolist(), all_patient_labels[cv_kfold_splits[fold_no][0]].tolist())
        val_data_and_labels = (all_patient_data[cv_kfold_splits[fold_no][1]].tolist(), all_patient_labels[cv_kfold_splits[fold_no][1]].tolist())
        
        # calculate class weights based on frequency of each class in training set
        class_counts = np.bincount(train_data_and_labels[1])
        total_samples = len(train_data_and_labels[1])
        class_weights = {0: total_samples / (2 * class_counts[0]), 1: total_samples / (2 * class_counts[1])}
        
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(train_data_and_labels[0],
                  train_data_and_labels[1],
                  epochs=num_epochs,
                  batch_size=batch_size,
                  verbose=1,
                  validation_data=val_data_and_labels,
                  class_weight=class_weights)

        # print metrics: sensitivity, specificity, confusion matrix, roc curve
        val_true = val_data_and_labels[1]
        val_pred = model.predict(val_data_and_labels[0])
        
        print(f"Sensitivity: {recall_score(val_true, np.round(val_pred))}")
        print(f"Specificity: {recall_score(val_true, np.round(val_pred), pos_label=0)}")

        confusion_matrix = ConfusionMatrixDisplay.from_predictions(val_true, np.round(val_pred))
        confusion_matrix.plot()
        plt.show()

        roc_curve = RocCurveDisplay.from_predictions(val_true, val_pred)
        roc_curve.plot()
        plt.show()

if __name__ == "__main__":
    if not COMPRESSED_FILES_CREATED:
        collect_data_and_kfolds()

    cross_validate()
