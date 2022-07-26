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

Run the generateImages script, the number of images you want to generate is given as a command
line argument (ensure you are within the project directory cloned from github)

    blenderproc run generateImages.py 1

Images and labels are saved to the outputTest directory. Images are saved as hdf5 files and can 
be viewd with blenderproc

    blenderproc vis hdf5 outputTest/0.hdf5
