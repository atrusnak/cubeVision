# cubeVision

## INFO

This project generates a sythetic dataset of Rubik's cube images 

The 3D cube models used are cube.blend and whiteCube.blend and they represent a black plastic and 
white plastic cube respectively. Images can be rendered with either by modifying the model path in generateImages.py
The models were origally obtained from Blenderkit https://www.blenderkit.com/

## Installation 

Clone the repository 

    git clone https://github.com/atrusnak/cubeVision

If using anaconda or similar virtual environment, first install pip

    conda install pip

From pip install blenderproc 
Blenderproc is the library that is used to help generate our synthetic dataset. For more info, see:
https://github.com/DLR-RM/BlenderProc

    pip install blenderproc

Run the generateImages script, the number of images you want to generate is given
as a command line argument (ensure you are within the project directory cloned 
from github)

Note: the first time blenderproc is run it will instal blender. This can take a few minutes.

    blenderproc run generateImages.py 1

Images and labels are saved to the outputTest directory. Images are saved as 
hdf5 files and can 
be viewed with blenderproc

    blenderproc vis hdf5 outputTest/0.hdf5

These hdf5 files can be converted into JPEG images using:

    python convertHDF5toJPG.py
    
Then the augment file will overwrite some of the generatedImages with color augmented verisons

    python augment.py


