import torch
import torchvision.transforms as transforms
from PIL import Image
import imgaug.augmenters as iaa
import os
import glob
import cv2
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.getcwd())

# Randomly augments the image to change a combination 
# of the brightness, contrast, saturation, and hue
# https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html

# Old method

#def augment(img):
#    transform = transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0)
#    img = transform(img)
#    return img

#https://github.com/aleju/imgaug 
# Chooses one of the effects and applies it to the image
one = iaa.OneOf([
    iaa.MedianBlur(k = (3,11)),
    iaa.AverageBlur(k = (2,7)),
    iaa.GaussianBlur(sigma=(0, 3.0))
])

# Accesses the first 20 images generated from generateImage.py, 
# augments them, and then saves them into a new directory
image_path = os.path.join(cur_path,"outputTest","data", "*.jpg")
images = [cv2.imread(image) for image in glob.glob(image_path)]
images_aug = one(images = images)
for i in range(20):
    filename = str(i).zfill(6)
    cv2.imwrite(os.path.join(cur_path, "outputTest", "augData", filename + ".jpg"), images_aug[i])