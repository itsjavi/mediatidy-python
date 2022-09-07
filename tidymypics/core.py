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


def predict(model, image_path, image_size, class_names, true_class=None):
    meta = fs.get_image_metadata(image_path)

    img_batch = load_img(image_path, image_size)
    # img_name = os.path.basename(image_path)

    predictions = model.predict(img_batch, verbose=0, workers=2)
    confidence_score = tf.nn.softmax(predictions[0])
    confidence_score_percent = 100 * np.max(confidence_score)
    pred_class = class_names[np.argmax(confidence_score)]

    meta['true_class'] = true_class
    meta['pred_class'] = pred_class
    meta['pred_confidence'] = confidence_score_percent

    return meta


def predict_dir(model, images_dir, class_names):
    image_files = fs.get_images_recursive(images_dir)
    results = []

    n_imgs = len(image_files)
    n_progress = 0

    ut.progress_bar(0, n_imgs)

    print(f"\n - Running predictions for all images under '{images_dir}'\n")

    for img_path in image_files:
        n_progress += 1
        results.append(predict(
            model=model,
            image_path=img_path,
            image_size=IMG_SIZE,
            class_names=class_names,
            true_class=None
        ))
        ut.progress_bar(n_progress, n_imgs)

    return pd.DataFrame(results)


def organize_images_dir(src, dest, by_year=True, move_files=False):
    output_dir = os.path.abspath(dest)
    verb = "Moved" if move_files == True else "Copied"

    # Scan src for images
    image_files = fs.get_images_recursive(src)

    if len(image_files) == 0:
        print("\nNo images found in the source folder.\n")
        return

    # Load model and classes
    print("\n\n--------------\n")
    (model, class_names) = load_model_from_disk(MODEL_NAME)
    print("--------------\n\n")
    print("Class Names: ", class_names, "\n")

    # Process file by file
    print("\n - Classifying and organizing images...\n")

    # dataset = []
    n_copied = 0
    duplicated = []
    n_imgs = len(image_files)
    n_progress = 0

    ut.progress_bar(0, n_imgs)

    for img_path in image_files:
        n_progress += 1

        try:
            img_meta = predict(
                model=model,
                image_path=img_path,
                image_size=IMG_SIZE,
                class_names=class_names,
                true_class=None
            )
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}, when reading file {img_path}")
            raise
        # dataset.append(img_meta)

        # -------------- Copy or move file  ----------------------
        md5code = img_meta['md5hash'][0:7]
        file_ext = pathlib.Path(img_path).suffix
        # file_folder = os.path.basename(os.path.dirname(img_path))

        if by_year:
            dest_path = os.path.join(
                output_dir, img_meta['pred_class'], img_meta['cyear']
                #output_dir, file_folder, img_meta['cyear']
            )
        else:
            dest_path = os.path.join(output_dir, img_meta['pred_class'])
            # dest_path = os.path.join(output_dir, file_folder)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        dest_file = os.path.join(
            dest_path, img_meta['cdate'] + '-' + md5code + file_ext
        )

        if not os.path.exists(dest_file):
            print(img_path)
            if move_files:
                shutil.move(img_path, dest_file)
            else:
                # "noop"
                shutil.copy2(img_path, dest_file)  # copy2 = copy with metadata
            n_copied += 1
        else:
            duplicated.append(img_path)
        # --------------------------------------------------------

        ut.progress_bar(n_progress, n_imgs)

    print(
        f"{verb} {n_copied}/{len(image_files)} images."
    )

    if len(duplicated) > 0:
        print(f'Not {verb} ({len(duplicated)}): ', duplicated)

    # return pd.DataFrame(dataset)


def train_test_model(
    dataset_dir,
    test_dir,
    img_size=IMG_SIZE,
    batch_size=32,
    validation_split=0.2,
    seed=None,
    epochs=10
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
        (model, _) = load_model_from_disk(MODEL_NAME)
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
        images_dir=test_dir, class_names=class_names, model=model
    )
    # print(test_pred_df.to_string())

    # TODO: return the classification report for the test predictions, instead of a DataFrame

    return (class_names, model_summary, model_history, classif_report, test_pred_df)
