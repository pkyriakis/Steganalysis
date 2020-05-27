import tensorflow as tf
import tensorflow.keras.layers as L
import os
from layers import *
from utils import *
import glob
import pandas as pd

IMG_SIZE = 512
BATCH_SIZE = 64


def make_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_type2=5, dropout_rate=0):
    # I reduced the size (image size, filters and depth) of the original network because it was way to big
    inputs = L.Input(shape=input_shape)
    # with tf.device("/GPU:0"):
    x = layer_type1(inputs, filters=64, kernel_size=(8, 8), stride=8, dropout_rate=0)
    x = layer_type1(x, filters=64, dropout_rate=0)

    # with tf.device("/GPU:1"):
    for _ in range(num_type2):
        x = layer_type2(x, filters=64, dropout_rate=dropout_rate)

        # with tf.device("/GPU:0"):
    x = layer_type3(x, filters=16, dropout_rate=dropout_rate)
    x = layer_type3(x, filters=32, dropout_rate=dropout_rate)
    x = layer_type3(x, filters=64, dropout_rate=dropout_rate)
    x = layer_type3(x, filters=128, dropout_rate=dropout_rate)

    # with tf.device("/GPU:1"):
    x = layer_type4(x, filters=256, dropout_rate=dropout_rate)
    x = L.Dense(256)(x)
    x = L.Dropout(0.4)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    if dropout_rate > 0:
        x = L.Dropout(dropout_rate)(x)

    predictions = L.Dense(4, activation=tf.keras.activations.softmax)(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[alaska_tf, 'acc'])

    return model


def prepare_for_training(ds, training=True, cache=True):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.

    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    # Repeat forever
    if training:
        ds = ds.repeat()

    ds = ds.batch(2 * BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=100)

    return ds


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = make_model(num_type2=5, dropout_rate=0)
model.summary()

list_ds = tf.data.Dataset.list_files(str("data/*/*"), seed=42, shuffle=True)
labeled_ds = list_ds.map(process_path, num_parallel_calls=32)
data_train, data_val = split_dataset(labeled_ds, 0.02)
train_ds = prepare_for_training(labeled_ds, cache="cache_train")
test_ds = prepare_for_training(data_val, cache="cache_test")

model.fit(train_ds, epochs=1, validation_data=test_ds, steps_per_epoch=200, verbose=1, validation_steps=100)
model.save_weights("pre_trained_on_diff")

model = tf.keras.models.load_model("pre_trained_on_diff")

test_filenames = sorted(glob.glob("Test/*.jpg"))
test_df = pd.DataFrame({'ImageFileName': list(test_filenames)}, columns=['ImageFileName'])

list_ds = tf.data.Dataset.list_files(str("Test/*"), seed=42, shuffle=False)
test_ds = list_ds.map(get_img, num_parallel_calls=32)
test_ds = test_ds.batch(64)
predictions = model.predict(test_ds)

preds = np.array(predictions)
labels = preds.argmax(axis=-1)
new_preds = np.zeros((preds.shape[0],))
new_preds[labels != 0] = preds[labels != 0, 1:].sum(axis=1)
new_preds[labels == 0] = preds[labels == 0, 0]

test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
test_df['Label'] = new_preds

test_df = test_df.drop('ImageFileName', axis=1)
test_df.to_csv('submission_srnet_dct_filter.csv', index=False)
print(test_df.head())
