import numpy as np
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
from keras.models import Sequential
import matplotlib.pyplot as plt
import csv


NUM_SAMPLES = 51
images = []
for i in range(NUM_SAMPLES):
    try:
        with h5py.File('outputCornerPrediction/' + str(i) + '.hdf5', 'r') as f:
            image = np.array(f['colors'])
            images.append(image)
    except:
        print("skipped" + str(i))




images = np.array(images,dtype='float32') / 255.0
print(images.shape)
print(np.max(images[0]))

labels = pd.read_csv('outputCornerPrediction/cornerPoints.csv', header=None).to_numpy()
labels = np.array(labels,dtype='float32') / 255.0

batch_size = 8
img_height = 256
img_width = 256
input_shape = (img_width,img_height,3)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = .10)

num_points = 4
mobileNet = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        )
mobileNet.trainable = False
model = keras.Sequential(
    [
        # keras.Input(shape=input_shape),
        # layers.Conv2D(32, kernel_size=(5, 5), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Flatten(),
        # layers.Dropout(0.3),
        mobileNet,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_points*2, activation="sigmoid"),
    ]
)


model.compile(optimizer='adam',
              loss=keras.losses.MeanSquaredError(),
              metrics=['mean_squared_error'])

"""## Train the model"""

epochs=10
history = model.fit(
    x_train,y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test)
)

model.summary()

"""## Visualize training results

Create plots of loss and accuracy on the training and validation sets:
"""

mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, mse, label='Training MSE')
plt.plot(epochs_range, val_mse, label='Validation MSE')
plt.legend(loc='lower right')
plt.title('Training and Validation MSE')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
print(x_test.shape)
predictedPoints = model(x_test)[0]
print(predictedPoints)
# with open("outputCornerPrediction/predictedCornerPoints.csv", 'a') as f:
#     writer = csv.writer(f)
#     for p in predictedPoints:
#         writer.writerow(p)

testLabel = predictedPoints*255
plt.imshow(x_test[0])
plt.plot(testLabel[0],256-testLabel[1], 'bo')
plt.plot(testLabel[2],256-testLabel[3], 'bo')
plt.plot(testLabel[4],256-testLabel[5], 'bo')
plt.plot(testLabel[6],256-testLabel[7], 'bo')
plt.show()
