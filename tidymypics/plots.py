import matplotlib.pyplot as plt
import numpy as np


def plot_model_results(history, epochs):
    # Visualizing results

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_image_samples(image_size, class_names, batch_ds, grid_size=3):
    dpi = 96
    dpiq = 96/4

    plt.figure(figsize=(image_size[1]/dpiq, image_size[0]/dpiq), dpi=dpi)

    # show a random set of images every time
    take_idx = np.random.choice(len(batch_ds))
    # print("len(batch_ds) =",len(batch_ds))

    for images, labels in batch_ds.take(take_idx):
        for i in range(grid_size*grid_size):
            ax = plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i].numpy().astype(
                "uint8"), aspect="auto", cmap="gray")
            plt.title(class_names[labels[i]])
            plt.axis("off")
