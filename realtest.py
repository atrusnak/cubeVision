import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

model = keras.load_model("MaskerV1")
with Image.open("realImage.png") as im:
    array = np.asarray(im)
    print(array.shape)


