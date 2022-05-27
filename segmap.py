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

images = []
for i in range(150):
    try:
        with h5py.File('outputWithPOIvariation/' + str(i) + '.hdf5', 'r') as f:
            image = np.array(f['colors'])
            images.append(image)
    except:
        print('skipped file')


segMaps = []
for i in range(150):
    try:
        with h5py.File('outputWithPOIvariation/' + str(i) + '.hdf5', 'r') as f:
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



# images = preprocess_input(images)


batch_size = 4
img_height = 512
img_width = 512
image_size = 512

x_train, x_test, y_train, y_test = train_test_split(images, segMaps, test_size = .20)

# print(x_train.shape)
# print(y_test.shape)

num_classes = 2

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
    
    outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# model.load_weights('UNET')



epochs=10
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

model.save_weights('UNET2')

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


plt.imshow(testImage)
plt.show()
plt.imshow(testTrueMask)
plt.show()
plt.imshow(testPredMask)

plt.show()


# """The plots show that training accuracy and validation accuracy are off by large margins, and the model has achieved only around 60% accuracy on the validation set.

# Let's inspect what went wrong and try to increase the overall performance of the model.

# ## Overfitting

# In the plots above, the training accuracy is increasing linearly over time, whereas validation accuracy stalls around 60% in the training process. Also, the difference in accuracy between training and validation accuracy is noticeable—a sign of [overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit).

# When there are a small number of training examples, the model sometimes learns from noises or unwanted details from training examples—to an extent that it negatively impacts the performance of the model on new examples. This phenomenon is known as overfitting. It means that the model will have a difficult time generalizing on a new dataset.

# There are multiple ways to fight overfitting in the training process. In this tutorial, you'll use *data augmentation* and add *Dropout* to your model.

# ## Data augmentation

# Overfitting generally occurs when there are a small number of training examples. [Data augmentation](./data_augmentation.ipynb) takes the approach of generating additional training data from your existing examples by augmenting them using random transformations that yield believable-looking images. This helps expose the model to more aspects of the data and generalize better.

# You will implement data augmentation using the following Keras preprocessing layers: `tf.keras.layers.RandomFlip`, `tf.keras.layers.RandomRotation`, and `tf.keras.layers.RandomZoom`. These can be included inside your model like other layers, and run on the GPU.
# """

# data_augmentation = keras.Sequential(
#   [
#     layers.RandomFlip("horizontal_and_vertical",
#                       input_shape=(img_height,
#                                   img_width,
#                                   3))
#   ]
# )

# """Let's visualize what a few augmented examples look like by applying data augmentation to the same image several times:"""

# # plt.figure(figsize=(10, 10))
# # for images, _ in train_ds.take(1):
# #   for i in range(9):
# #     augmented_images = data_augmentation(images)
# #     ax = plt.subplot(3, 3, i + 1)
# #     plt.imshow(augmented_images[0].numpy().astype("uint8"))
# #     plt.axis("off")

# """You will use data augmentation to train a model in a moment.

# ## Dropout

# Another technique to reduce overfitting is to introduce [dropout](https://developers.google.com/machine-learning/glossary#dropout_regularization) regularization to the network.

# When you apply dropout to a layer, it randomly drops out (by setting the activation to zero) a number of output units from the layer during the training process. Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.

# Let's create a new neural network with `tf.keras.layers.Dropout` before training it using the augmented images:
# """
# base_model = tf.keras.applications.MobileNetV2(input_shape=(500,500,3), include_top=False, weights='imagenet')
# base_model.trainable = False

# model = Sequential([
#   data_augmentation,
#   layers.Rescaling(1./255),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   # layers.Conv2D(64, 3, padding='same', activation='relu'),
#   # layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

# """## Compile and train the model"""

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.summary()

# epochs = 15
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs,
#   batch_size=2
# )

# """## Visualize training results

# After applying data augmentation and `tf.keras.layers.Dropout`, there is less overfitting than before, and training and validation accuracy are closer aligned:
# """

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# """## Predict on new data

# Finally, let's use our model to classify an image that wasn't included in the training or validation sets.

# Note: Data augmentation and dropout layers are inactive at inference time.
# """

# # sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# # sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# # img = tf.keras.utils.load_img(
# #     sunflower_path, target_size=(img_height, img_width)
# # )
# # img_array = tf.keras.utils.img_to_array(img)
# # img_array = tf.expand_dims(img_array, 0) # Create a batch

# # predictions = model.predict(img_array)
# # score = tf.nn.softmax(predictions[0])

# # print(
# #     "This image most likely belongs to {} with a {:.2f} percent confidence."
# #     .format(class_names[np.argmax(score)], 100 * np.max(score))
# # )
