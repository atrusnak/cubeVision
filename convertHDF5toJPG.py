import h5py 
import numpy as np
from PIL import Image

for i in range(300):
    hdf = h5py.File('outputTest/'+str(i)+".hdf5",'r')
    array = hdf["colors"][:]
    img = Image.fromarray(array.astype('uint8'), 'RGB')
    filename = str(i).zfill(6)
    img.save("outputTest/"+"jpgs/"+filename+".jpg", "JPEG")
