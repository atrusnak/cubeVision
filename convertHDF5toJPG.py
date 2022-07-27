import h5py 
import numpy as np
from PIL import Image
import os

cur_path = os.path.abspath(os.getcwd())
for i in range(100):
    hdf = h5py.File(cur_path + '\outputTest\\\\'+ "" + str(i+10)+".hdf5",'r')
    array = hdf["colors"][:]
    img = Image.fromarray(array.astype('uint8'), 'RGB')
    filename = str(i).zfill(6)
    img.save(cur_path + "\outputTest"+"\data"+filename+".jpg", "JPEG")
