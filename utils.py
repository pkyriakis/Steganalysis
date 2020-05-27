from sklearn import metrics
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 512


def alaska_wuac_metric(y_true, y_pred):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    y_true = np.array(y_true)
    y_true[y_true != 0] = 1

    y_pred = np.array(y_pred)
    labels = y_pred.argmax(axis=1)
    temp = y_pred[labels != 0, 1:]
    new_preds = np.zeros((len(y_pred),))
    new_preds[labels != 0] = temp.sum(axis=1)
    new_preds[labels == 0] = y_pred[labels == 0, 0]
    y_pred = new_preds

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    competition_metric = 0

    try:
        for idx, weight in enumerate(weights):
            y_min = tpr_thresholds[idx]
            y_max = tpr_thresholds[idx + 1]
            mask = (y_min < tpr) & (tpr < y_max)

            x_padding = np.linspace(fpr[mask][-1], 1, 100)

            x = np.concatenate([fpr[mask], x_padding])
            y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
            y = y - y_min  # normalize such that curve starts at y=0
            score = metrics.auc(x, y)
            submetric = score * weight
            best_subscore = (y_max - y_min) * weight
            competition_metric += submetric
    except:
        return 0.5

    competition_metric = competition_metric / normalization
    return competition_metric


def alaska_tf(y_true, y_val):
    """Wrapper for the above function"""
    return tf.py_function(func=alaska_wuac_metric, inp=[y_true, y_val], Tout=tf.float32)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # label = tf.cast(parts[-2] == "Cover", tf.int8)
    label = 0
    if parts[-2] == 'JUNIWARD':
        label = 1
    if parts[-2] == 'UERD':
        label = 2
    if parts[-2] == 'JMiPOD':
        label = 3
    return label


def get_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
    return img


def process_path(file_path):
    # Get embeeded image
    label = get_label(file_path)
    img = get_img(file_path)

    # Get cover
    parts = tf.strings.split(file_path, os.path.sep)
    cover_file = parts[-1]
    cover = get_img('data/Cover/' + cover_file)

    # Return their difference
    return tf.subtract(img, cover), label


def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset
