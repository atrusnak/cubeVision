import torch
import torchvision.transforms as transforms
from PIL import Image
#import imgaug.augmenters as iaa
import os

cur_path = os.path.abspath(os.getcwd())

# Randomly augments the image to change a combination 
# of the brightness, contrast, saturation, and hue
# https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html

def augment(img):
    transform = transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.30)
    img = transform(img)
    return img

# An additional image augmentation library with increased effects and augmentations
#https://github.com/aleju/imgaug 
#one = iaa.OneOf([
#    iaa.MedianBlur(k = (3,11)),
#    iaa.AverageBlur(k = (2,7)),
#    iaa.GaussianBlur(sigma=(0, 3.0))
#])

# Accesses the first 60 images generated from generateImage.py, 
# augments them, and then overwrites the image with the new augmented version
for i in range(60):
    filename = str(i).zfill(6)
    image_path = cur_path + "\outputTest\\\\" + "data" + filename+ ".jpg"
    image = Image.open(image_path)
    image = augment(image)
    image = image.save(image_path)