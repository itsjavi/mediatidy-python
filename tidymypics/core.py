import os
import json
import pathlib
import shutil
import logging
import warnings
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report as sk_classification_report
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
from . import fs
from . import utils as ut

# Docs about color mode, image size, loading images etc:
# - https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img
# - https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

IMG_SIZE = (800, 480)
COLOR_MODE = "grayscale"  # grayscale (1 channel), rgb (3ch), rgba (4ch)
COLOR_CHANNELS = 1
MODELS_PATH = os.path.join(PROJECT_DIR, 'models')
MODEL_NAME = 'tidymypics-model'
MODEL_FILE = f'{MODELS_PATH}/{MODEL_NAME}.h5'


def ignore_warnings():
    warnings.filterwarnings("ignore")


def disable_tf_logger():
    # show only tensorflow errors, hide warnings and debug messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(1)


def versions():
    return {"tf": tf.__version__, "keras": tf.keras.__version__}


def train_validation_split(
    # root directory of the subdirectories (one per class), containing the images
    images_dir,
    image_size,
    batch_size=32,
    validation_split=0.2,
    seed=123
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
            layers.RandomFlip("horizontal_and_vertical",
                              input_shape=input_shape),
            layers.GaussianNoise(0.01, input_shape=input_shape)
        ]
    )

    model = Sequential([
        data_augmentation,
        # Normalize colors from 0-255 to 0.0-1.0
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(filters=16, kernel_size=3,
                      padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=32, kernel_size=3,
                      padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=64, kernel_size=3,
                      padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),  # droput, to reduce overfitting
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


def train_model(model, ds, epochs=10):
    (class_names, train_ds, validation_ds) = ds
    return model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs
    )


def load_img(path, image_size):
    img = tf.keras.utils.load_img(
        path, target_size=image_size, color_mode=COLOR_MODE
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    return img_array


def predict(model, image_path, image_size, class_names, true_class=None):
    img_batch = load_img(image_path, image_size)
    img_name = os.path.basename(image_path)

    predictions = model.predict(img_batch, verbose=0, workers=2)
    score = tf.nn.softmax(predictions[0])
    score_percent = 100 * np.max(score)
    pred_class = class_names[np.argmax(score)]

    # if (score_percent < score_threshold):
    #     verdict_class = low_score_class_map[verdict_class]

    return {
        "name": img_name,
        "pred_class": pred_class,
        "pred_confidence": score_percent,
        "true_class": true_class,
        "path": image_path
    }


def predict_dir(model, images_dir, image_size, class_names):
    pred_data_dir = pathlib.Path(images_dir)
    pred_images = list(pred_data_dir.glob('**/*.jpg')) + \
        list(pred_data_dir.glob('**/*.png'))
    results = []

    for im in pred_images:
        img_path = os.path.abspath(str(im))

        if "ipynb_checkpoints" in img_path:
            continue

        results.append(
            predict(model, img_path, image_size, class_names)
        )

    return pd.DataFrame(results)


def predict_df(model, images_df, image_size, class_names):
    print(" - Predicting the class of every image...")

    imgc = len(images_df)
    imgn = 0

    ut.progress_bar(0, imgc)

    for index, im in images_df.iterrows():
        imgn += 1
        pred = predict(model, im['path'], image_size, class_names)
        images_df.loc[index, 'pred_class'] = pred['pred_class']
        images_df.loc[index, 'pred_confidence'] = pred['pred_confidence']
        ut.progress_bar(imgn, imgc)

    return images_df


def save_model(model, name, class_names):
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    # model.save('models/model01') # Tensorflow SavedModel format (many files)
    model.save(f'{MODELS_PATH}/{name}.h5')  # Keras H5 format (single file)
    with open(f'{MODELS_PATH}/{name}-class_names.json', 'w') as f:
        json.dump(class_names, f)


def model_exists(name):
    f = f'{MODELS_PATH}/{name}.h5'
    return os.path.exists(f)


def load_model_from_disk(name):
    # print(f'{MODELS_PATH}/{name}.h5')
    model = tf.keras.models.load_model(f'{MODELS_PATH}/{name}.h5')
    with open(f'{MODELS_PATH}/{name}-class_names.json', 'r') as f:
        class_names = json.load(f)

    return (model, class_names)


def get_classification_report(model, ds):
    (class_names, train_ds, validation_ds) = ds
    true_categories = tf.concat([y for x, y in validation_ds], axis=0).numpy()

    y_pred = model.predict(validation_ds, verbose=0, workers=2)
    predicted_categories = np.argmax(y_pred, axis=1)

    return sk_classification_report(true_categories, predicted_categories, target_names=class_names)


def organize_images_dir(src, dest, by_year=True, move_files=False):
    imgfiles = fs.get_images_recursive(src)
    imgdata = fs.get_images_metadata(imgfiles)

    print("\n\n--------------\n")
    (model, class_names) = load_model_from_disk(MODEL_NAME)
    print("--------------\n\n")

    print("Class Names: ", class_names, "\n")
    #print("Model Summary:")
    # print(model.summary())

    img_size = IMG_SIZE
    imgdata['pred_class'] = None
    imgdata['pred_confidence'] = None

    # TODO: read metadata and predict one by one, to have a single progress bar

    predictions_df = predict_df(
        model, imgdata, image_size=img_size, class_names=class_names
    )

    output_dir = os.path.abspath(dest)

    copied = 0
    not_copied = 0
    not_copied_paths = []

    verb = "Moved" if move_files == True else "Copied"

    if move_files:
        print("\n - Moving images...\n")
    else:
        print("\n - Copying images...\n")

    for index, row in predictions_df.iterrows():
        src_file = row['path']
        md5code = row['md5hash'][0:7]

        file_ext = pathlib.Path(src_file).suffix

        if by_year:
            dest_path = os.path.join(
                output_dir, row['pred_class'], row['cyear']
            )
        else:
            dest_path = os.path.join(output_dir, row['pred_class'])

        dest_file = os.path.join(
            dest_path, row['cdate'] + '-' + md5code + file_ext)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        if not os.path.exists(dest_file):
            if move_files:
                shutil.move(src_file, dest_file)
            else:
                # "noop"
                shutil.copy2(src_file, dest_file)  # copy2 = copy with metadata
            copied += 1
        else:
            not_copied += 1
            not_copied_paths.append(src_file)

    print(
        f"{verb} {copied}/{len(predictions_df)} images."
    )

    if not_copied > 0:
        print(f'Not {verb} ({not_copied}): ', not_copied_paths)


def train_test_model(
    dataset_dir,
    test_dir,
    img_size=IMG_SIZE,
    batch_size=32,
    validation_split=0.2,
    seed=None,
    epochs=15
):
    # create train/validation dataset split, detecting classes
    ds = ds_optimize(
        train_validation_split(
            images_dir=dataset_dir,
            image_size=img_size,
            batch_size=batch_size,
            validation_split=validation_split,
            seed=seed
        )
    )

    class_names = ds[0]

    # load existing model if exists, to train it with new data
    if model_exists(MODEL_NAME):
        print(
            f" - Model '{MODEL_NAME}.h5' already exists, loading it to train with new data."
        )
        model = load_model_from_disk(MODEL_NAME)
    else:
        # compile and build the new model
        print(" - Creating a new model...")
        model = build_model(ds=ds, image_size=img_size)
        model_summary = model.summary()

    # train model
    model_history = train_model(model, ds, epochs=epochs)

    # generate classif. report
    classif_report = get_classification_report(model, ds)

    # persist model
    save_model(model, MODEL_NAME, class_names)

    # predict with new unseen data
    test_pred_df = predict_dir(
        images_dir=test_dir, class_names=class_names, image_size=img_size, model=model
    )
    # print(test_pred_df.to_string())

    # TODO: return the classification report for the test predictions, instead of a DataFrame

    return (class_names, model_summary, model_history, classif_report, test_pred_df)
