# tidyphotos
Command-line app written in Python to classify your photo library by year,
separating the screenshots from the actual photos using a trained Tensorflow model.

This project consists of 2 parts:
- Python notebooks to create and test the model
- Python CLI script to organize the images using that model


## Requirements

- UNIX-compatible OS (only tested on macOS 12)
- Python 3.8+
- PIP 22.0+
- Tensorflow 2.9+
    - For macOS, follow [this guide](https://developer.apple.com/metal/tensorflow-plugin/)


Dataset required to train the model:

- Images correctly classified under the `data/train/photo` and `data/train/screenshot` directories of this project.
- At least 1000 images per class (photo or screenshot).
- The more and the more diverse, the better.
- Both classes should contain the same number of images (to avoid bias).
- Avoid duplicated images.


## Usage

### Setup

- Clone or download this repository
- Open your terminal and change the directory to this one
- Run `pip install -r requirements.txt` to install the dependencies

### Creating a new model

To create a new model with your own data (pictures), use the `notebooks/model-builder.ipynb` notebook and run
all the steps. You can use Jupyter, Jupyter Lab, VS Code or any other compatible IDE to run it.

If you want to test your model, you can use the `notebooks/model-tester.ipynb` notebook.
For that, you will need images under `data/test/photo` and `data/test/screenshot`.


### Using the Command-Line tool
Once you have a model, you can use the command-line tool to organize your images.
From the source code folder, run something like this:

```bash
# Usage: python -m tidyphotos SOURCE_DIR OUTPUT_DIR
#   Example:
python -m tidyphotos ~/Pictures/MyMessyAlbums ~/Pictures/OrganizedAlbums
``` 

All the images will be organized by class (photo or screenshot) and year when the picture was taken.
Files will be also renamed using the creation timestamp, and the first 7 characters of the image's MD5 hash
to avoid duplicate file names.

Images will be copied (preserving metadata), so be sure you have enough space on your hard drive to have a duplicate
of the source directory.


## Other classifications

You can change your classification class names to others (and even have more than 2), 
but take in consideration the model architecture and layers:

- Adds random noise
- Flips images randomly (either vertically or horizontally)
- Reads the images as grayscale
- Reads the images resizing them to 800x480px (or the size specified in the notebooks)
- Uses a Drop Out layer
(check tidyphotos/core.py to know more)

So, with the built-in (pre)processing you may lose important information depending on your classification needs (like the color).
