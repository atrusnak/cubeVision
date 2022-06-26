import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from numpy.lib.function_base import append
import tensorflow as tf
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from keras.applications.efficientnet_v2 import EfficientNetV2L
from keras.applications.efficientnet_v2 import preprocess_input
from matplotlib import pyplot as plt

numImages = 20
images = []
for i in range(numImages):
    try:
        with h5py.File('outputStickerSeg/' + str(i) + '.hdf5', 'r') as f:
            image = np.array(f['colors'])
            images.append(image)
    except:
        print('skipped file')


segMaps = []
for i in range(numImages):
    try:
        with h5py.File('outputStickerSeg/' + str(i) + '.hdf5', 'r') as f:
            segMap= np.array(f['class_segmaps'])
            segMaps.append(segMap)
    except:
        print('skipped file')


images = np.array(images,dtype='float32') / 255.0
segMaps = np.array(segMaps,dtype='float32')
print(images.shape)
print(segMaps.shape)
print(np.max(images[0]))
print(np.max(segMaps[0]))

segMapChannels = np.zeros((numImages,128,128,10))
for m in range(numImages):
    for i in range(128):
        for j in range(128):
            classIndex = segMaps[m][i,j] 
            segMapChannels[m][i][j][int(classIndex)]=1
            

print(segMapChannels.shape)
print(segMapChannels[0])


# images = preprocess_input(images)


batch_size = 2
img_height = 128
img_width = 128
image_size = 128

x_train, x_test, y_train, y_test = train_test_split(images, segMapChannels, test_size = .20)

# print(x_train.shape)
# print(y_test.shape)

num_classes = 10

# https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# model.load_weights('UNET')



epochs=5
history = model.fit(
    x_train,y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    # callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=.0001)
 )

print(history.history.keys())
"""## Visualize training results

Create plots of loss and accuracy on the training and validation sets:
"""

# model.save_weights('UNET2')
#model.save("MaskerV2")

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

testImage = x_test[0]
testTrueMask = y_test[0]
testPredMasks = model.predict(x_test)
testPredMask = testPredMasks[0]
print(testPredMask.shape)

def maskToImage(mask):
    image = np.zeros((128,128))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for c in range(mask.shape[2]):
                if mask[i,j,c] == 1:
                    image[i,j]=c
    return image/10.0
            

print(np.sum(testTrueMask))
print(np.sum(testPredMask))


plt.imshow(testImage)
plt.show()
plt.imshow(maskToImage(testTrueMask))
plt.show()
plt.imshow(maskToImage(testPredMask))

plt.show()
