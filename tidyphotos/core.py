import warnings
import matplotlib.pyplot as plt
import pathlib
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import re
from sklearn.metrics import classification_report as sk_classification_report
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image, ExifTags
from datetime import datetime

# Docs about color mode, image size, loading images etc:
# - https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img
# - https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

COLOR_MODE = "grayscale" # grayscale (1 channel), rgb (3ch), rgba (4ch)
COLOR_CHANNELS = 1
MODELS_PATH = os.path.join(PROJECT_DIR, 'models')

def ignore_warnings():
    warnings.filterwarnings("ignore")

def versions():
    return {"tf": tf.__version__, "keras": tf.keras.__version__}


def train_validation_split(
    images_dir, # root directory of the subdirectories (one per class), containing the images
    image_size, 
    batch_size = 32, 
    validation_split = 0.2, 
    seed = 123
):
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        images_dir,
        validation_split=validation_split,
        subset="training",
        color_mode=COLOR_MODE,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )
    
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        images_dir,
        validation_split=validation_split,
        subset="validation",
        color_mode=COLOR_MODE,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )
    
    return (train_ds.class_names, train_ds, validation_ds)

# TODO add prepare_images function (resize, convert to grayscale, create variants for training (zoom, rotate, negative, etc))

def ds_optimize(ds):
    (class_names, train_ds, validation_ds) = ds
    
    # autotune for better performance (memory buffering). shuffle train_ds as well.
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return (class_names, train_ds, validation_ds)



def build_model(ds, image_size):
    (class_names, train_ds, validation_ds) = ds
    input_shape = (image_size[0], image_size[1], COLOR_CHANNELS)
    num_classes = len(class_names)
    
    # --- LAYERS -----------------------------------------------------------------------------
    
    # data augmentation, to reduce overfitting
    data_augmentation = Sequential(
      [
        layers.RandomFlip("horizontal_and_vertical", input_shape=input_shape),
        layers.GaussianNoise(0.01, input_shape=input_shape)
      ]
    )
    
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=input_shape), # Normalize colors from 0-255 to 0.0-1.0
        layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2), # droput, to reduce overfitting
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        # output layer:
        layers.Dense(units=num_classes, name="outputs")
    ])
    
    # ---------------------------------------------------------------------------------
    # compile and build:
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.build(input_shape=input_shape)
        
    
    return model


def train_model(model, ds, epochs = 10):
    (class_names, train_ds, validation_ds) = ds
    return model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs
    )


def load_img(path, image_size):
    img = tf.keras.utils.load_img(path, target_size=image_size, color_mode=COLOR_MODE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    return img_array


def predict(model, image_path, image_size, class_names, low_score_class_map, score_threshold = 70):
    img_batch = load_img(image_path, image_size)
    img_name = os.path.basename(image_path)

    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0])
    score_percent = 100 * np.max(score)
    pred_class = class_names[np.argmax(score)]
    verdict_class = pred_class
    
    if (score_percent < score_threshold):
        verdict_class = low_score_class_map[verdict_class]

    return {
        "name": img_name,
        "pred_class": pred_class,
        "pred_confidence": score_percent,
        "verdict_class": verdict_class,
        "path": image_path
    }


def predict_dir(model, images_dir, image_size, class_names, low_score_class_map, score_threshold = 70):
    pred_data_dir = pathlib.Path(images_dir)
    pred_images = list(pred_data_dir.glob('**/*.jpg')) + list(pred_data_dir.glob('**/*.png'))
    results = []
    
    for im in pred_images:
        img_path = os.path.abspath(str(im))
        
        if "ipynb_checkpoints" in img_path:
            continue
        
        results.append(
            predict(model, img_path, image_size, class_names, low_score_class_map, score_threshold)
        )


    return pd.DataFrame(results)

def predict_df(model, images_df, image_size, class_names, low_score_class_map, score_threshold = 70):
    for index, im in images_df.iterrows():
        pred = predict(model, im['path'], image_size, class_names, low_score_class_map, score_threshold)
        images_df.loc[index, 'pred_class'] = pred['pred_class']
        images_df.loc[index, 'pred_confidence'] = pred['pred_confidence']
        
    return images_df


def save_model(model, name, class_names):
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    
    #model.save('models/model01') # Tensorflow SavedModel format (many files)
    model.save(f'{MODELS_PATH}/{name}.h5') # Keras H5 format (single file)
    with open(f'{MODELS_PATH}/{name}-class_names.json', 'w') as f:
        json.dump(class_names, f)
        
def load_model_from_disk(name):
    print(f'{MODELS_PATH}/{name}.h5')
    model = tf.keras.models.load_model(f'{MODELS_PATH}/{name}.h5')
    with open(f'{MODELS_PATH}/{name}-class_names.json', 'r') as f:
        class_names = json.load(f)
        
    return (model, class_names)


def get_classification_report(model, ds):
    (class_names, train_ds, validation_ds) = ds
    true_categories = tf.concat([y for x, y in validation_ds], axis = 0).numpy() 

    y_pred = model.predict(validation_ds)
    predicted_categories = np.argmax(y_pred, axis = 1)

    return sk_classification_report(true_categories, predicted_categories, target_names=class_names)
