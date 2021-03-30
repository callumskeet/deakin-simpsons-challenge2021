import os
import itertools
import collections
import datetime
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import wandb
from wandb.keras import WandbCallback

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    classification_report,
)

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import models, layers, optimizers
from tensorflow.python.keras.saving import hdf5_format
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorboard.plugins.hparams import api as hp


class ModelCheckpoint_tweaked(tf.keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        **kwargs,
    ):

        # Change tf_utils source package.
        from tensorflow.python.keras.utils import tf_utils

        super().__init__(
            filepath,
            monitor,
            verbose,
            save_best_only,
            save_weights_only,
            mode,
            save_freq,
            **kwargs,
        )


def init_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def create_datasets(data_params):
    # Now, we create a training data iterator by creating batchs of images of the same size as
    # defined previously, i.e., each image is resized in a 64x64 pixels format.
    train_ds = DirectoryIterator(
        **data_params,
        subset="training",
    )

    # Similarly, we create a validation data iterator by creating batchs of images of the same size as
    # defined previously, i.e., each image is resized in a 64x64 pixels format.
    val_ds = DirectoryIterator(**data_params, subset="validation", shuffle=False)

    test_data_params = data_params.copy()
    test_data_params["image_data_generator"] = ImageDataGenerator()
    test_data_params["directory"] = "dataset/simpsons_test"
    test_ds = DirectoryIterator(**test_data_params)

    # We save the list of classes (labels).
    class_names = list(train_ds.class_indices.keys())

    # We also save the number of labels.
    num_classes = train_ds.num_classes

    return train_ds, val_ds, test_ds, class_names, num_classes


def plot_class_distribution(train_ds, class_names):
    """Plots the distribution of examples for each class"""
    counter = collections.Counter(train_ds.labels)
    v = [[class_names[item[0]], item[1]] for item in counter.items()]
    df = pd.DataFrame(data=v, columns=["index", "value"])

    g = sns.catplot(
        x="index",
        y="value",
        data=df,
        kind="bar",
        legend=False,
        height=4,
        aspect=4,
        saturation=1,
    )
    g.despine(top=False, right=False)

    plt.xlabel("Classes")
    plt.ylabel("#images")
    plt.title("Distribution of images per class")
    plt.xticks(rotation="vertical")
    plt.show()
    return g


def plot_data(train_ds, class_names):
    """Displays 30 images from the training set"""
    plt.figure(figsize=(20, 16))
    images = []
    labels = []
    for itr in train_ds.next():
        for i in range(30):
            if len(images) < 30:
                images.append(itr[i].astype("uint8"))
            else:
                labels.append(list(itr[i]).index(1))

    for i in range(len(images)):
        ax = plt.subplot(5, 6, i + 1)
        plt.imshow(images[i])
        plt.title(
            class_names[labels[i]].replace("_", " ") + " (" + str(int(labels[i])) + ")"
        )
        plt.axis("off")
