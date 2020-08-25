import keras
import numpy as np
from keras import backend as k
from keras.applications import imagenet_utils
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.metrics import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def prepare_image(file):
    img_path = ""
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255

    if show:
        plt.imshow(img_tensor[0])
        plt.axis("off")
        plt.show()

    return img_tensor


if __name__ == "__main__":
    mobile = keras.applications.mobilenet.MobileNet()
    base_model = MobileNet(weights="imagenet", include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    preds = Dense(10, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=preds)

    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_directory(
        directory="data",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True
    )

    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    step_size_train = train_generator.n // train_generator.batch_size
    model.fit(train_generator, steps_per_epoch=step_size_train, epochs=10)

    # SavedModel.
    # The model architecture, and training configuration (including the optimizer, losses, and metrics)
    # are stored in `saved_model.pb`. The weights are saved in the `variables/` directory.
    model.save("model_mbnet")

    # Demo.
    new_image = load_image("data/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
    prediction = model.predict(new_image)
    print(prediction)
