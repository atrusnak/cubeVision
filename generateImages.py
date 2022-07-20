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

# cube coloring functions
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


def generateImages(outputDirectory):
    bproc.init()
    # load rubik's cube model (already positioned at (0,0,0))
    bproc.loader.load_blend("cube.blend")

    # collect objects as bpy objects (as opposed to blenderproc)
    objects = bpy.data.objects
    faceCubes = []
    for obj in objects:
        if('FRONT' in obj.name):
            faceCubes.append(obj)


    # Randomly scramble the front cube face

    # record the scramble as a 1D vector
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

    # pretty sure their filter shit just doesn't work
    # faceCubes = bproc.filter.by_attr(cubes,"name", "FRONT*", regex=True)

    # get face cubes and assign them to a different category for image segmentation later
    bprocFaceCubes = []
    for i, cube in enumerate(bprocCubes):
        if('FRONT' in cube.get_name()):
            print(cube.get_name())
            bprocFaceCubes.append(cube)
            

    for i, faceCube in enumerate(bprocFaceCubes):
            faceCube.set_cp("category_id",scramble[i]+1)
    # reset scramble after using it for id's
    scramble = []


    imageSize=256
    bproc.camera.set_resolution(256,256)



    # this could be useful later
    # light now being controlled by hdri alone

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



    #sample camera positions
    location = np.random.uniform([0.1, 0.5, 0.1], [-0.1, 0.2, -0.1])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(np.random.uniform([0.05, 0.05, 0.05], [-0.05, -0.05, -0.05]) - location)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


    camera = bpy.data.objects['Camera']
    scene = bpy.context.scene




    # Determine corner points in world coordinates
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


    tr = mathutils.Vector((-halfCubeLength, halfCubeLength, halfCubeLength))
    tl = mathutils.Vector((halfCubeLength, halfCubeLength, halfCubeLength))
    bl = mathutils.Vector((halfCubeLength, halfCubeLength, -halfCubeLength))
    br = mathutils.Vector((-halfCubeLength, halfCubeLength, -halfCubeLength))

    leftTranslate = np.array((thirdCubeLength,0,0))
    upTranslate = np.array((0,0,thirdCubeLength))

    brCorner = np.array((-halfCubeLength, halfCubeLength, -halfCubeLength))

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
    worldPoints = [b1,b2,b3,b4,mb1,mb2,mb3,mb4,mt1,mt2,mt3,mt4,t1,t2,t3,t4]

    




    
    cameraPoints = []
    pixelPoints = []
    for p in worldPoints:
        cameraPoint = bpy_extras.object_utils.world_to_camera_view(scene,camera,mathutils.Vector(p))
        cameraPoints.append(cameraPoint)

        pixelPoints.extend([int(n) for n in ((cameraPoint*imageSize).to_tuple(0))[:-1]])

    print(cameraPoints)
    print(pixelPoints)

    with open("outputTest/cornerPoints.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(pixelPoints)











    bproc.renderer.set_noise_threshold(0.01)

# set random hdri background and lighting
    haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven("haven")
    bproc.world.set_world_background_hdr_img(haven_hdri_path)

    # with open(os.path.join(outputDirectory, "havenCheck", 'a')) as f:
    #     writer = csv.writer(f)
    #     writer.writerow(haven_hdri_path)

    data = bproc.renderer.render()

# for segmentation map
    # data.update(bproc.renderer.render_segmap(map_by=["instance", "class"]))


# write to file
    bproc.writer.write_hdf5("outputTest", data, append_to_existing_output=True)

# Write data to coco file
    # bproc.writer.write_coco_annotations(outputDirectory,
    #                                 instance_segmaps=data["instance_segmaps"],
    #                                 instance_attribute_maps=data["instance_attribute_maps"],
    #                                 colors=data["colors"],
    #                                 color_file_format="JPEG",
    #                                 append_to_existing_output=True)


def main():
    numImages = int(sys.argv[1])
    directories = ["training", "test", "validation"]
    for d in directories:
        for i in range(numImages):
            try:
                generateImages(os.path.join("dataset/", d))
            except Exception as e:
                print("Unresolved blenderproc error, continueing")
                i-=1
                # with open("outputTest/havenCheck", 'a') as f:
                #     writer = csv.writer(f)
                #     writer.writerow("Error!")
            bproc.utility.reset_keyframes()
if __name__ == '__main__':
    main()
