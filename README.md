# cubeVision

## Installation 

Clone the repository 

    git clone https://github.com/atrusnak/cubeVision

Switch to peer review branch

    git checkout for_peer_review

If using anaconda or similar virtual environment, first install pip

    conda install pip

From pip install blenderproc

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

## Peer Review

The code to be reviewed is found in the file **augment.py** in the root 
directory of the repository. This file color augments the specified amount of images. 
I am using the simple ColorJitter function from torchvision, I will be looking into a 
more sophfisticated version of this augmentation which keeps track of the keypoints and
bounding boxes, so I can introduce positional augmentation as well.
