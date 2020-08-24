import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle
from tqdm import tqdm

from main import centering_image
from main import create_dataframe
from translate import translate as t

if __name__ == "__main__":
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
    name_animal = []
    for i in range(10):
        path = train_df[train_df.category == i].values[2]
        name_animal.append(t.get(path[0].split("/")[-2]))

    model = load_model("model/ft.h5")
    test_images = []
    j = 20  # change this to get different images
    for i in range(10):
        path = train_df[train_df.category == i].values[j]
        a = images[train_df[train_df.filename == path[0]].index[0]]
        img = np.array(a)
        img = img[:, :, ::-1].copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] > img.shape[1]:
            tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
        else:
            tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))
        img = centering_image(cv2.resize(img, dsize=tile_size))
        img = img[16:240, 16:240]
        test_images.append(img)

    test_images = np.array(test_images).reshape(-1, 224, 224, 3)
    something = model.predict(test_images)
    animals = name_animal
    i = 0
    for pred in something:
        path = train_df[train_df.category == i].values[2]
        plt.imshow(test_images[i])
        print(f"\nActual  : {animals[i]}")
        print(f"Predict : {animals[np.where(pred.max() == pred)[0][0]]}")
        plt.show()
        i += 1
