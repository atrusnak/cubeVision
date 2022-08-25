import blenderproc as bproc
import numpy as np
import bpy
import bpy_extras  
import math
import random
import csv
import mathutils
import os
import sys
import json

# cube coloring functions
# We can change the color displayed on the front face of the Rubik's cube by rotating 
# the colored cubes that make up the full Rubik's cube
# The rotation necessary to display each color is made into a function below

# These functions take a face cube as input and rotate it to the appopriate color
def red(cube):
    cube.rotation_euler = (math.radians(0), math.radians(0),math.radians(90))

def blue(cube):
    cube.rotation_euler =  (math.radians(0), math.radians(0),math.radians(0))

def orange(cube):
    cube.rotation_euler =  (math.radians(0), math.radians(0),math.radians(-90))

def green(cube):
    cube.rotation_euler =  (math.radians(0), math.radians(0),math.radians(180))

def yellow(cube):
    cube.rotation_euler =  (math.radians(90), math.radians(0),math.radians(0))

def white(cube):
    cube.rotation_euler =  (math.radians(-90), math.radians(0),math.radians(0))


#The main function to generate an image of a random cube
def generateImages(outputDirectory, imageID):
    #need to initialize blenderproc
    bproc.init()
    # load rubik's cube model (already positioned at (0,0,0))
    # here we can also load whiteCube.blend for a white plastic cube
    bproc.loader.load_blend("cube.blend")

    # collect objects as bpy objects (as opposed to blenderproc)
    objects = bpy.data.objects
    # the face cubes are the 9 colored cubes that make up the front face
    faceCubes = []
    for obj in objects:
        if('FRONT' in obj.name):
            faceCubes.append(obj)


    # Randomly scramble the front cube face

    # record the scramble as a 1D vector
    # each color is represented as a number between 0-5
    scramble = []
    faceCubes.reverse()
    for cube in faceCubes:
        print(cube.name)
        color = random.randint(0,5)
        scramble.append(color)
        if(color == 0):
            red(cube)
        if(color == 1):
            blue(cube)
        if(color == 2):
            orange(cube)
        if(color == 3):
            green(cube)
        if(color == 4):
            yellow(cube)
        if(color == 5):
            white(cube)
    print(scramble)
    # with open("outputTest/scrambles.csv", 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(scramble)



    # can compute point of interest if you want to point the camera. Just used origin tho
    # poi = bproc.object.compute_poi(c)

    # collect objects as blenderproc objects
    bprocCubes = bproc.object.get_all_mesh_objects()

    # DEPRECATED
    # get face cubes and assign them to a different category for image segmentation later
    bprocFaceCubes = []
    for i, cube in enumerate(bprocCubes):
        if('FRONT' in cube.get_name()):
            print(cube.get_name())
            bprocFaceCubes.append(cube)
            

    for i, faceCube in enumerate(bprocFaceCubes):
            faceCube.set_cp("category_id",scramble[i]+1)



    #set image render size as 256 square
    imageSize=256
    bproc.camera.set_resolution(imageSize,imageSize)



    # light now being controlled by hdri alone
    # This is used in high lighting variance augmentation

    # light = bproc.types.Light()
    # light.set_type("POINT")
    # light.set_location(
    #     bproc.sampler.sphere(
    #         center=[0,0,0],
    #         radius=1,
    #         mode='SURFACE'
    #     )
    # )
    # light.set_energy(1000)



    #sample random camera positions
    location = np.random.uniform([0.1, 0.5, 0.1], [-0.1, 0.2, -0.1])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(np.random.uniform([0.05, 0.05, 0.05], [-0.05, -0.05, -0.05]) - location)
    # build transformation matrix between world and camera coordinates
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    # add the camera to the world
    bproc.camera.add_camera_pose(cam2world_matrix)


    # Collect the camera and scene objects
    camera = bpy.data.objects['Camera']
    scene = bpy.context.scene




    # Determine corner points in world coordinates of each sticker
    # units are meters
    # center of cube is 0,0,0
    # Width of cube is 0.05, use 0.025 for half
    # tr = top right, bl = bottom left etc
    # camera is in the positive y direction meaning:
    # up : +z
    # left : +x
    # in (toward camera) : +y
    halfCubeLength = 0.025
    cubeLength = halfCubeLength*2
    thirdCubeLength = cubeLength/3.0

    #adding leftTranslate and upTranslate to a point will translate them one sticker length left 
    # or up respectively, this is used to calculate all the sticker corner points relative to the bottom right
    # corner
    leftTranslate = np.array((thirdCubeLength,0,0))
    upTranslate = np.array((0,0,thirdCubeLength))

    #starting point is the bottom right corner
    brCorner = np.array((-halfCubeLength, halfCubeLength, -halfCubeLength))

    #calculate all 16 key sitcker corner points using the defined translations 

    """
    The key points are the corners of the stickers, defined like so 
    t4 = top four
    mt1 = middle top one etc

    t4--t3--t2--t1
     |   |   |   | 
    mt4--mt3--mt2--mt1
     |   |   |   | 
    mb4--mb3--mb2--mb1
     |   |   |   | 
    b4--b3--b2--b1




    """
    b1 = brCorner
    b2 = np.add(b1,leftTranslate).tolist()
    b3 = np.add(b2,leftTranslate).tolist()
    b4 = np.add(b3,leftTranslate).tolist()
    mb1 = np.add(b1,upTranslate).tolist()
    mb2 = np.add(mb1,leftTranslate).tolist()
    mb3 = np.add(mb2,leftTranslate).tolist()
    mb4 = np.add(mb3,leftTranslate).tolist()
    mt1 = np.add(mb1,upTranslate).tolist()
    mt2 = np.add(mt1,leftTranslate).tolist()
    mt3 = np.add(mt2,leftTranslate).tolist()
    mt4 = np.add(mt3,leftTranslate).tolist()
    t1 = np.add(mt1,upTranslate).tolist()
    t2 = np.add(t1,leftTranslate).tolist()
    t3 = np.add(t2,leftTranslate).tolist()
    t4 = np.add(t3,leftTranslate).tolist()


    #These points are defined in the world coordinate system, we must transfrom them 
    # to the camera coordinate system to use them for bounding boxes
    worldPoints = [b1,b2,b3,b4,mb1,mb2,mb3,mb4,mt1,mt2,mt3,mt4,t1,t2,t3,t4]




    
    #create lists for the desired points
    cameraPoints = []
    #pixel points define the points in pixels instead of a float between 0 and 1
    pixelPoints = []
    for p in worldPoints:
        #convert the world points to camera points
        cameraPoint = bpy_extras.object_utils.world_to_camera_view(scene,camera,mathutils.Vector(p))
        #don't need a depth dimension
        cameraPoint[1] = 1-cameraPoint[1]
        cameraPoints.append(cameraPoint)

        #pixel points define the points in pixels
        pixelPoints.extend([int(n) for n in ((cameraPoint*imageSize).to_tuple(0))[:-1]])


    # with open("outputTest/cornerPoints.csv", 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(pixelPoints)



    # sticker1Corners = [0, 1, 4, 5]
    # sticker2Corners = [1, 2, 5, 6]
    # sticker3Corners = [2, 3, 6, 7]
    # sticker4Corners = [4, 5, 8, 9]
    # sticker5Corners = [5, 6, 9, 10]
    # sticker6Corners = [6, 7, 10, 11]
    # sticker7Corners = [8, 9, 12, 13]
    # sticker8Corners = [9, 10, 13, 14]
    # sticker9Corners = [10, 11, 14, 15]
    # cubeCorners = [0, 3, 14, 15]
    # stickers = np.array([sticker1Corners,sticker2Corners,sticker3Corners,sticker4Corners,sticker5Corners,sticker6Corners,sticker7Corners,sticker8Corners,sticker9Corners])

    #arrang the pixel points into groups by bounding box. Grouping the four corners
    # of each sticker together
    print(pixelPoints)
    stickers = np.array([[0, 1, 4, 5],
                    [1, 2, 5, 6],
                    [2, 3, 6, 7],
                    [4, 5, 8, 9],
                    [5, 6, 9, 10],
                    [6, 7, 10, 11],
                    [8, 9, 12, 13],
                    [9, 10, 13, 14],
                    [10, 11, 14, 15],
                    [0, 3, 12, 15]])

    

    
    # Every rendered image needs to store label information
    # we will return this info in annotations and images

    annotations = []
    images = []
    for c in range(stickers.shape[0]):
        #by seperating out the x and y values we can more easily determine the bounding boxes
        xValues = []
        yValues = []
        for i in stickers[c]:
            xValues.append(pixelPoints[i*2])
            yValues.append(pixelPoints[(i*2)+1])

        #the last set of points is for the whole cube not a sticker and must be labeled differently
        if(c == stickers.shape[0]-1):
            annotID = (imageID*10)+c
            annot = generateAnnotations(annotID, imageID, 6, xValues, yValues)
            print("cube" + str(c))
        #stickers get labeled with their respective color
        else:
            print("sticker" + str(c))
            annotID = (imageID*10)+c
            annot = generateAnnotations(annotID, imageID, scramble[c], xValues, yValues)




        annotations.append(annot)


    #following the COCO format each image needs an id and file name
    imageDict = {
            'id':imageID,
            'file_name':str(imageID).zfill(6)+'.jpg',
            'width':256,
            'height':256
            }
    images.append(imageDict)


    #noise threshold determines the accuracy of the render
    bproc.renderer.set_noise_threshold(0.01)
    cur_path = os.path.abspath(os.getcwd())
    # set random hdri background and lighting
    # some hdri's from haven will cause an unresolvable error that must be caught and handled later
    try:
        haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(os.path.join(cur_path, "haven"))
        bproc.world.set_world_background_hdr_img(haven_hdri_path)
    except Exception as e:
        raise 

   # with open(os.path.join('outputTest', "havenCheck"), 'a') as f:
   #     f.write(haven_hdri_path+'\n')



    #the final images are rendered 
    data = bproc.renderer.render()

# for segmentation map
# NO LONGER USED
    # data.update(bproc.renderer.render_segmap(map_by=["instance", "class"]))


# write the image to file
    bproc.writer.write_hdf5("outputTest", data, append_to_existing_output=True)

# Write data to coco file
# NO LONGER USED
    # bproc.writer.write_coco_annotations('outputTest',
    #                                 instance_segmaps=data["instance_segmaps"],
    #                                 colors=data["colors"],
    #                                 color_file_format="JPEG",
    #                                 append_to_existing_output=True)

    return annotations, images

#The main function called from command line to generate images
def main():
    #The basic structure of the labels json is defined below
    labelDict = {
            'info': {
                'dataset_name': 'cubeVision',
                'authors': 'Alex and Keegan'
                },
            'categories':[
                    {
                        'id':0,
                        'name':'red',
                        'supercategory':'sticker'
                    },
                    {
                        'id':1,
                        'name':'blue',
                        'supercategory':'sticker'
                    },
                    {
                        'id':2,
                        'name':'orange',
                        'supercategory':'sticker'
                    },
                    {
                        'id':3,
                        'name':'green',
                        'supercategory':'sticker'
                    },
                    {
                        'id':4,
                        'name':'yellow',
                        'supercategory':'sticker'
                    },
                    {
                        'id':5,
                        'name':'white',
                        'supercategory':'sticker'
                    },
                    {
                        'id':6,
                        'name':'cube',
                        'supercategory':'wholeCube'
                    }
                ]
            }
    #determine the number of images to render from cli arg
    numImages = int(sys.argv[1])
    allAnnotations = []
    allImages = []
    cur_path = os.path.abspath(os.getcwd())
    for i in range(numImages):
        #try to render each image, some configurations may fail so we catch errors
        try:
            bproc.utility.reset_keyframes()
            annotations, images = generateImages("dataset/", i)

        except Exception as e:
            print("Unresolved blenderproc error, continueing")
            #upon failure we want to decrement our counter so the desired number of images is rendered
            i-=1
            #print diagnostics on which background failed to render
            with open(os.path.join(cur_path + '\outputTest', "havenCheck"), 'a') as f:
                f.write("ERROR!")
            continue
        #collect all images and annotations from renders
        allAnnotations.extend(annotations)
        allImages.extend(images)

    labelDict["annotations"] = allAnnotations
    labelDict["images"] = allImages
    #write the labels to the labels.json file
    annotJsonString = json.dumps(labelDict)
    with open('outputTest/labels.json', 'w+') as labelFile:
        labelFile.write(annotJsonString)


#This function return the appropriate JSON section for one bounding box annotation
#This function's args:
#the ID of the bounding box
#the ID of the image
#the categoryID (color) of the sticker
#the x and y values of the corner points
# return a dictionary matching the json format for COCO labels
def generateAnnotations(ID, imageID, categoryID, xValues, yValues):
    #determine the top left and top right corners
    x = int(min(xValues))
    y = int(min(yValues))
    width = int(np.abs(x-max(xValues)))
    height = int(np.abs(y-max(yValues)))
    #bounding boxes are defined as the top left corner and their height+width
    bbox = [x,y,width,height]
    annotationDict = {
            'id':ID,
            'image_id':imageID,
            'category_id':categoryID,
            'bbox':bbox
            }
    return annotationDict


if __name__ == '__main__':
    #can be used for testing because normal runs won't output errors
    # generateImages('outputTest', 0)
    main()
