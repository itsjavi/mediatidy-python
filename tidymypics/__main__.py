import shlex
import sys
import getopt
import os
import pathlib
import shutil
import hashlib
from . import fs
from . import core as co

ORGANIZE_MODE = "organize"
TRAIN_MODE = "train"
PREDICT_MODE = "predict"


def print_err(phrase: str) -> None:
    print("ERROR: " + phrase)


def get_validated_args():
    help_text = """
    
    =====================
     TidyMyPics 1.0.0
    =====================

        Image Classifier and Organizer, by Javier Aguilar (itsjavi.com).
        
        Given a directory of images, analyizes it recursively,
        separating the photos from the screenshot-like images,
        and organizes them by year and date.

        This CLI tool can also be used to train a new Deep Learning model using Tensorflow
        > Read how in https://github.com/itsjavi/tidymypics#readme

    USAGE:

    python -m tidymypics [-h|--help|-t|--train] <SOURCE_PATH> [<DEST_PATH>] [...]
        
        OPTIONS:

        -h | --help:    Prints this help screen.
    
        -t | --train:   Activates the model training mode.
                        
                        Uses the SOURCE_PATH as train dataset to train a new model,
                        and the DEST_PATH to test it.

                        As a result, it will print how accurate your new model is for
                        each class.

                        Both directories should not be your real photo library, but
                        pictures classified by hand in sub-directories named after each
                        class (e.g. "photo", "screenshot", etc.).

                        Read more about this in the project README.md file.
        
        -p | --predict: Predicts the class of one single image or more. You should specify
                        the image path(s) as arguments.

                        The output will be printed as CSV, with the following data:
                        predicted class, confidence score, year, date, shorhand md5 hash
                        
                        e.g.: 
                            class,score,year,date,hash
                            screenshot,96.5,2021,20210618-182005,2332a92


        ARGUMENTS:
            
        <SOURCE_PATH>:  REQUIRED. The root directory where your (unorganized) pictures are stored.
                        
                        In model training mode, this will be your dataset and each sub-directory
                        should be a different class name containing hand-classified pictures
                        (for the final model, have at least 1000 per class).

        <DEST_PATH>:    The destination directory where to copy your organized pictures library.
                        
                        In model training mode, you should have hand-classified pictures to test
                        your model with, in a similar way as the train dataset. This is needed
                        to verify that the model predicted unseen pictures correctly.

                        This argument is not required when you use the `-p` or `--pic` option.

    """

    mode = ORGANIZE_MODE
    _args = sys.argv
    del _args[0]  # first arg is always current file

    if len(_args) == 0:
        print_err(help_text)
        sys.exit(0)

    try:
        opts, args = getopt.getopt(
            _args, "ht:p:", ["help", "train", "predict"])
    except getopt.GetoptError:
        print_err(help_text)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_text)
            sys.exit(0)
        elif opt in ("-t", "--train"):
            mode = TRAIN_MODE
        elif opt in ("-t", "--train"):
            mode = PREDICT_MODE

    if mode != PREDICT_MODE and len(args) != 2:
        print_err("This app needs exactly 2 arguments: SOURCE_PATH and DEST_PATH.")
        sys.exit(1)

    if mode == PREDICT_MODE and len(args) < 1:
        print_err(
            "You need at least one image path as argument to start predicting.")
        sys.exit(1)

    src_dir = os.path.abspath(args[0])
    dest_dir = os.path.abspath(args[1])

    if not os.path.isdir(src_dir) or len(src_dir) == 0:
        # Both modes should have a valid SOURCE_PATH
        print(
            f"SOURCE_PATH ERROR: '{src_dir}' is not a valid existing directory."
        )
        sys.exit(1)

    if mode == TRAIN_MODE and not os.path.isdir(dest_dir) or len(dest_dir) == 0:
        # Training mode should have an existing DEST_PATH
        print(
            f"DEST_PATH ERROR: '{dest_dir}' is not a valid existing directory."
        )
        sys.exit(1)

    if mode == ORGANIZE_MODE and os.path.isdir(dest_dir):
        # attempting to use an existing dir to output the organized photo library in
        print(
            f"DEST_PATH ERROR: '{dest_dir}' already exists and cannot be used as output directory."
        )
        sys.exit(1)

    return {
        "src_dir": src_dir,
        "dest_dir": dest_dir,
        "paths": args,
        "mode": mode
    }


def main() -> int:
    args = get_validated_args()
    print(args)

    (src_dir, DEST_PATH) = args

    if args['training_mode'] == True:
        print("TRAINING...")
    else:
        co.organize_images_dir(args['src_dir'], args['dest_dir'])

    return 0


if __name__ == '__main__':
    sys.exit(main())
