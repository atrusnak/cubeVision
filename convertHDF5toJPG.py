import h5py 
import numpy as np
from PIL import Image


for i in range(3200):
    hdf = h5py.File('outputTest/'+str(i+10)+".hdf5",'r')
    array = hdf["colors"][:]
    img = Image.fromarray(array.astype('uint8'), 'RGB')
    filename = str(i).zfill(6)
    img.save("outputTest/"+"data/"+filename+".jpg", "JPEG")
