import os
from datetime import datetime
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflowjs as tfjs
from keras import applications
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.utils import shuffle
from tqdm import tqdm

from translate import translate as t


def create_dataframe():
    classes = [
        "elefante",
        "farfalla",
        "mucca",
        "cavallo",
        "pecora",
        "ragno",
        "cane",
        "gallina",
        "gatto",
        "scoiattolo",
    ]
    categories = []
    files = []

    for path in glob(os.path.join("data", "**", "*.*"), recursive=True):
        files.append(path)
        categories.append(classes.index(path.split("/")[-2]))

    df = pd.DataFrame({"filename": files, "category": categories})
    train_df = pd.DataFrame(columns=["filename", "category"])
    for i in range(10):
        train_df = train_df.append(df[df.category == i].iloc[:500, :])

    train_df.head()
    train_df = train_df.reset_index(drop=True)
    train_df.to_csv("animals.csv")
    return train_df


def centering_image(img):
    size = [256, 256]
    img_size = img.shape[:2]

    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row : (row + img.shape[0]), col : (col + img.shape[1])] = img

    return resized


def show_plots(hist):
    """Useful function to view plot of loss values & accuracies across the various epochs"""
    loss_vals = hist["loss"]
    val_loss_vals = hist["val_loss"]
    epochs = range(1, len(hist["accuracy"]) + 1)

    f, ax = plt.subplot(nrows=1, ncols=2, figsize=(16, 4))

    # plot losses on ax[0]
    ax[0].plot(
        epochs,
        loss_vals,
        color="navy",
        marker="o",
        linestyle=" ",
        label="Training Loss",
    )
    ax[0].plot(
        epochs, val_loss_vals, color="firebrick", marker="*", label="Validation Loss"
    )
    ax[0].set_title("Training & Validation Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend(loc="best")
    ax[0].grid(True)

    # plot accuracies
    acc_vals = history["accuracy"]
    val_acc_vals = history["val_accuracy"]

    ax[1].plot(
        epochs, acc_vals, color="navy", marker="o", ls=" ", label="Training Accuracy"
    )
    ax[1].plot(
        epochs, val_acc_vals, color="firebrick", marker="*", label="Validation Accuracy"
    )
    ax[1].set_title("Training & Validation Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend(loc="best")
    ax[1].grid(True)

    plt.show()
    plt.close()

    # delete locals from heap before exiting
    del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals


if __name__ == "__main__":
    begin = datetime.now().replace(microsecond=0)
    print(f"Begin: {begin}")

    train_df = create_dataframe()
    y = train_df["category"]
    x = train_df["filename"]

    x, y = shuffle(x, y, random_state=8)

    images = []
    with tqdm(total=len(train_df)) as pbar:
        for i, file_path in enumerate(train_df.filename.values):
            # read image
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize
            if img.shape[0] > img.shape[1]:
                tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
            else:
                tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

            # centering
            img = centering_image(cv2.resize(img, dsize=tile_size))

            # output 224x224px
            img = img[16:240, 16:240]
            images.append(img)
            pbar.update(1)

    images = np.array(images)

    rows, cols = 2, 5
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20))
    for i in range(10):
        path = train_df[train_df.category == i].values[2]
        # image = cv2.imread(path[0])/
        axes[i // cols, i % cols].set_title(path[0].split("/")[-2] + str(path[1]))
        axes[i // cols, i % cols].imshow(
            images[train_df[train_df.filename == path[0]].index[0]]
        )

    data_num = len(y)
    random_index = np.random.permutation(data_num)

    x_shuffle = []
    y_shuffle = []
    for i in range(data_num):
        x_shuffle.append(images[random_index[i]])
        y_shuffle.append(y[random_index[i]])

    x = np.array(x_shuffle)
    y = np.array(y_shuffle)
    val_split_num = int(round(0.2 * len(y)))
    x_train = x[val_split_num:]
    y_train = y[val_split_num:]
    x_test = x[:val_split_num]
    y_test = y[:val_split_num]

    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    img_rows, img_cols, img_channel = 224, 224, 3
    name_animal = []
    for i in range(10):
        path = train_df[train_df.category == i].values[2]
        name_animal.append(t.get(path[0].split("/")[-2]))

    base_model = applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(img_rows, img_cols, img_channel),
    )

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation="relu"))
    add_model.add(Dense(10, activation="softmax"))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=["accuracy"],
    )
    model.summary()

    batch_size = 32
    epochs = 50

    train_datagen = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.1, horizontal_flip=True
    )
    train_datagen.fit(x_train)

    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[
            ModelCheckpoint("VGG16-transferlearning.model", monitor="val_acc"),
            EarlyStopping(patience=10, restore_best_weights=True),
        ],
    )

    # SavedModel.
    # The model architecture, and training configuration (including the optimizer, losses, and metrics)
    # are stored in `saved_model.pb`. The weights are saved in the `variables/` directory.
    model.save("model")

    # Keras also supports saving a single HDF5 file containing the model's architecture, weights values,
    # and `compile()` information. It is a light=weight alternative to SavedModel.
    model.save("model/ft.h5")

    # The TensorFlow.js converter has two components:
    # 1. A command line utility that converts Keras and TensorFlow models for use in TensorFlow.js.
    # `$ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model`
    # 2. An API for loading and execute the model in the browser with TensorFlow.js.
    tfjs.converters.save_keras_model(
        model, "target", weight_shard_size_bytes=1024 * 1024 * 5
    )

    end = datetime.now().replace(microsecond=0)
    print(f"\n-> End: {end}")
    print(f"\n\n--> Elapsed time: {end - begin}")
