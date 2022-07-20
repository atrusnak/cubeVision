import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt


# max of 436
NUM_SAMPLES = 3
images = []
for i in range(NUM_SAMPLES):
    try:
        with h5py.File('outputTest/' + str(i) + '.hdf5', 'r') as f:
            image = np.array(f['colors'])
            images.append(image)
    except: print("skipped" + str(i))


images = np.array(images,dtype='float32')/255.0



predictions = pd.read_csv('outputTest/cornerPoints.csv', header=None).to_numpy()


batch_size = 8
img_height = 256
img_width = 256




for i in range(len(images)):
    testImage = images[i]
    testLabel = predictions[i]

    plt.imshow(testImage)
    plt.plot(testLabel[0],256-testLabel[1], 'bo')
    plt.plot(testLabel[2],256-testLabel[3], 'bo')
    plt.plot(testLabel[4],256-testLabel[5], 'bo')
    plt.plot(testLabel[6],256-testLabel[7], 'bo')
    plt.plot(testLabel[8],256-testLabel[9], 'bo')
    plt.plot(testLabel[10],256-testLabel[11], 'bo')
    plt.plot(testLabel[12],256-testLabel[13], 'bo')
    plt.plot(testLabel[14],256-testLabel[15], 'bo')
    plt.plot(testLabel[16],256-testLabel[17], 'bo')
    plt.plot(testLabel[18],256-testLabel[19], 'bo')
    plt.plot(testLabel[20],256-testLabel[21], 'bo')
    plt.plot(testLabel[22],256-testLabel[23], 'bo')
    plt.plot(testLabel[24],256-testLabel[25], 'bo')
    plt.plot(testLabel[26],256-testLabel[27], 'bo')
    plt.plot(testLabel[28],256-testLabel[29], 'bo')
    plt.plot(testLabel[30],256-testLabel[31], 'bo')
    plt.show()
