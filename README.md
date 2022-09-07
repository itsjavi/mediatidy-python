# tidymypics
Command-line app written in Python to classify your photo library by year,
separating different image classes (of your choice) using a trained Tensorflow model.

This CLI application consists of 2 parts:
- A command to generate and train a new model
- A command to classify your photo library and organize it by class and year


## Requirements

- UNIX-compatible OS (only tested on macOS 12)
- Python 3.8+ with PIP 22.0+
- Tensorflow 2.9+
    - For macOS, follow [this guide](https://developer.apple.com/metal/tensorflow-plugin/)


Dataset required to train the model:

- Images correctly classified manually under the `data/train` directory of this project. One folder for every different class, for example: `data/train/photo` and `data/train/screenshot`.
- At least 1000 images (samples) for every class class.
- The more and the more diverse, the better.
- All classes should contain the same number of images (to avoid bias), or aproximately.
- Avoid duplicated images. Try not to add images that are too similar (e.g. photo bursts).


## Usage

### Setup

- Clone or download this repository
- Open your terminal and change the directory to this one
- Run `pip install -r requirements.txt` to install the dependencies

### Creating a new model

To create a new model with your own data (pictures), use the `python -m tidymypics --train TRAIN_PATH TEST_PATH`.

A new model will be generated using `TRAIN_PATH` as the path where your train samples are stored (by class like mentioned before).

It also requires a `TEST_PATH` argument, which should be a directory similarly structured as `TRAIN_PATH`, 
to test your model with new and "unseen" images, and see how good are the predictions.


### Using the Command-Line tool
Once you have a model, you can use again the command-line tool to organize your images.
From the source code folder, run something like this:

```bash
# Usage: python -m tidymypics SOURCE_DIR OUTPUT_DIR
#   Example:
python -m tidymypics --year ~/Pictures/MyMessyAlbums ~/Pictures/OrganizedAlbums
``` 

All the images will be organized by class and year of when the picture was taken.
Files will be also renamed using the creation timestamp, and the first 7 characters of the image's MD5 hash
to avoid duplicate file names.

If you don't add the `--year` option, pictures won't be grouped by year, but they will all live in the same
directory, separated only by class.

If you add the `--move` option, pictures will be moved instead of copied.

> To know more about all the options and the usage, just run `python -m tidymypics` without any parameter.


## Implementation Considerations

You can use anything as your classification class names  (e.g. people, pet, landscape, artwork, text, nudity, etc.), 
but take in consideration the model architecture and layers:

- Adds random noise
- Flips images randomly (either vertically or horizontally)
- Reads the images as grayscale. If color information is important for your classification, you should modify the code and use `rgb` with 3 channels.
- Reads the images resizing them to 800x480px
- Uses a Drop Out layer

These transformations are only done in-memory, to compute the model, and not to the original images.

> Check `tidymypics/core.py`, `build_model()` function to know more details about the implementation.
