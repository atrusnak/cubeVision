import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt


# max of 436
NUM_SAMPLES = 10
images = []
for i in range(NUM_SAMPLES):
    try:
        with h5py.File('outputCornerPrediction/' + str(i) + '.hdf5', 'r') as f:
            image = np.array(f['colors'])
            images.append(image)
    except: print("skipped" + str(i))


images = np.array(images,dtype='float32')/255.0



predictions = pd.read_csv('outputTest/predictedCornerPoints.csv', header=None).to_numpy()


batch_size = 8
img_height = 256
img_width = 256




for i in range(10):
    testImage = images[i]
    testLabel = predictions[i]

    plt.imshow(testImage)
    plt.plot(testLabel[0],256-testLabel[1], 'bo')
    plt.plot(testLabel[2],256-testLabel[3], 'bo')
    plt.plot(testLabel[4],256-testLabel[5], 'bo')
    plt.plot(testLabel[6],256-testLabel[7], 'bo')
    plt.show()
