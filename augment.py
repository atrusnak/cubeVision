import torch
import torchvision.transforms as transforms
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import os
import glob
import cv2
import matplotlib.pyplot as plt
import json
from numpy import asarray
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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

# augments the image and keeps track of the bounding box changes
def augment(img, boxCoord):

    # 10 bounding boxes, one bounding box for each sticker and one bounding box for cube face
    b1 = boxCoord[0]
    b2 = boxCoord[1]
    b3 = boxCoord[2]
    b4 = boxCoord[3]
    b5 = boxCoord[4]
    b6 = boxCoord[5]
    b7 = boxCoord[6]
    b8 = boxCoord[7]
    b9 = boxCoord[8]
    b10 = boxCoord[9]

    # The generated json file from generateImages.py records the bounding boxes' lower left corner, width, and height
    # so x2 and y2 are just x2 = x1 + width and y2 = y1 + height 
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=b1[0], y1=b1[1], x2=b1[0]+b1[2], y2=b1[1]+b1[3]),
        BoundingBox(x1=b2[0], y1=b2[1], x2=b2[0]+b2[2], y2=b2[1]+b2[3]),
        BoundingBox(x1=b3[0], y1=b3[1], x2=b3[0]+b3[2], y2=b3[1]+b3[3]),
        BoundingBox(x1=b4[0], y1=b4[1], x2=b4[0]+b4[2], y2=b4[1]+b4[3]),
        BoundingBox(x1=b5[0], y1=b5[1], x2=b5[0]+b5[2], y2=b5[1]+b5[3]),
        BoundingBox(x1=b6[0], y1=b6[1], x2=b6[0]+b6[2], y2=b6[1]+b6[3]),
        BoundingBox(x1=b7[0], y1=b7[1], x2=b7[0]+b7[2], y2=b7[1]+b7[3]),
        BoundingBox(x1=b8[0], y1=b8[1], x2=b8[0]+b8[2], y2=b8[1]+b8[3]),
        BoundingBox(x1=b9[0], y1=b9[1], x2=b9[0]+b9[2], y2=b9[1]+b9[3]),
        BoundingBox(x1=b10[0], y1=b10[1], x2=b10[0]+b10[2], y2=b10[1]+b10[3])
    ], shape=img.shape)

    # Will happen 50% of the time when called
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Sequential augmentation (The augmentation will happen in this order everytime)
    seq = iaa.Sequential([
        # 20% of the images will be flipped horizontally and 20% of the images will be flipped vertically
        iaa.Fliplr(.2),
        iaa.Flipud(.2),
        # One of the blur effects will be applied 50% of the time
        sometimes(
            iaa.OneOf([
                iaa.MedianBlur(k = (3,11)),
                iaa.AverageBlur(k = (2,7)),
                iaa.GaussianBlur(sigma=(0, 3.0)),
                iaa.BilateralBlur(d=(3,5)),
                iaa.MotionBlur(k = (3,7)),
                iaa.MeanShiftBlur()
            ])
        ),
        # Apply affine transformations to the images
        # translate by -20 to +20 relative to height/width (per axis)
        # rotate by -10 to +10 degrees
        # scale to 80-120% of image height/width (each axis independently)
        iaa.Affine(translate_percent = (-.2,.2),
                   rotate = [-10,10],
                   scale = (.5,1.5)
        ),
        sometimes(
            iaa.AddToBrightness((-30,30))
            )
    ])
    image_aug, bbs_aug = seq(image = img, bounding_boxes = bbs)

    # To add the bounding box to the images 

    # Rescale image and bounding boxes
    #image_rescaled = ia.imresize_single_image(img, (512, 512))
    #bbs_rescaled = bbs.on(image_rescaled)
    # Draw image before/after rescaling and with rescaled bounding boxes
    #image_bbs = bbs.draw_on_image(img)
    #image_aug = bbs_aug.draw_on_image(image_aug)

    return img, image_aug, bbs_aug

# Read in the generated labels json file
with open(os.path.join(cur_path, "outputTest","labels.json"),"r") as f:
    data = json.loads(f.read())
    bboxes = []
    aug_images = []
    annotations = data["annotations"]
    images = data["images"]
    i = 0
    # Loop through all images in the json file
    while i < len(annotations):
        bbox = []
        # Seperate the bounding boxes out by 10, every image should have 10
        for j in range(10):
            bbox.append(annotations[i]["bbox"])
            i = i + 1
        bboxes.append(bbox)
    # Rename the jpg files for the new json file 
    for m in range(len(images)):
        filename = str(m).zfill(6)
        images[m]["file_name"] = "a" + filename + ".jpg"
count = 0
for k in range(len(bboxes)):
    filename = str(k).zfill(6)
    image = Image.open(os.path.join(cur_path, "outputTest", "data", filename + ".jpg"))
    numpydata = asarray(image)
    image_bbs_data, image_aug_bbs_data, bbs_data = augment(numpydata, bboxes[k])
    imageBB = Image.fromarray(image_bbs_data)
    image_rescale_BB = Image.fromarray(image_aug_bbs_data)
    for i in range(10):
        aug_bbox = []
        aug_bbox.append(bbs_data[i].x1_int)
        aug_bbox.append(bbs_data[i].y1_int)
        aug_bbox.append(int(bbs_data[i].width))
        aug_bbox.append(int(bbs_data[i].height))
        annotations[count]["bbox"] = aug_bbox
        count = count + 1
    imageBB = imageBB.save(os.path.join(cur_path, "outputTest", "augDataBB", "a"+filename + ".jpg"))
    image_rescale_BB = image_rescale_BB.save(os.path.join(cur_path, "outputTest", "augData", "a"+filename + ".jpg"))
with open(os.path.join(cur_path, "outputTest","aug_labels.json"),"w") as f:
    f.write(json.dumps(data))