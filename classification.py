from keras.layers.core.activation import Activation
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
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

images = []
for i in range(436):
    with h5py.File('output/' + str(i) + '.hdf5', 'r') as f:
        image = np.array(f['colors'])
        images.append(image)


images = np.array(images,dtype='float32')



labels = pd.read_csv('output/scrambles.csv', header=None).to_numpy()
oneHotLabels = []
for l in range(len(labels)):
    oneHot = np.zeros(54)
    label = labels[l]
    for i in range(len(label)):
        thisInt = label[i]
        oneHot[(i*6)+thisInt] = 1
    oneHotLabels.append(oneHot)
        
oneHotLabels = np.array(oneHotLabels)

# images = preprocess_input(images)


batch_size = 8
img_height = 512
img_width = 512

x_train, x_test, y_train, y_test = train_test_split(images, oneHotLabels, test_size = .10)

print(x_train.shape)
print(y_test.shape)

num_classes = 54

base_model = VGG16(input_shape=(512,512,3), weights='imagenet', include_top=False, pooling='avg')

for layer in base_model.layers:
    layer.trainable=False

model = Sequential([
    layers.Rescaling(1./255),
    # base_model,
    layers.Conv2D(32,3, strides=2,padding='same'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64,3, strides=2,padding='same'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    # layers.Dense(512, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='sigmoid')
])

"""## Compile the model

For this tutorial, choose the `tf.keras.optimizers.Adam` optimizer and `tf.keras.losses.SparseCategoricalCrossentropy` loss function. To view training and validation accuracy for each training epoch, pass the `metrics` argument to `Model.compile`.
"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

"""## Model summary

View all the layers of the network using the model's `Model.summary` method:
"""


"""## Train the model"""

epochs=40
history = model.fit(
    x_train,y_train,
    epochs=epochs,
    batch_size = batch_size
)

model.summary()
"""## Visualize training results

Create plots of loss and accuracy on the training and validation sets:
"""

acc = history.history['accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
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
