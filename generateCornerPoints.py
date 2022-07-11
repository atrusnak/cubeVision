import blenderproc as bproc
import numpy as np
import bpy
import bpy_extras  
import math
import random
import csv
import mathutils


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


def generateImages():
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
    with open("outputTest/scrambles.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(scramble)



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
        # else:
        #     cube.set_cp("category_id", 7)
            

    colorDict = {}
    usedColors = []
    counter = 0
    newScramble = list(range(9))
    for i, c in enumerate(scramble):
        if c not in usedColors:
            usedColors.append(c)
            counter+=1
            colorDict[c] = counter
            newScramble[i] = counter
        else:
            newScramble[i] = colorDict[c]

    

    print(newScramble)

    for i, faceCube in enumerate(bprocFaceCubes):
            faceCube.set_cp("category_id",newScramble[i])
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

    tr = mathutils.Vector((-halfCubeLength, halfCubeLength, halfCubeLength))
    tl = mathutils.Vector((halfCubeLength, halfCubeLength, halfCubeLength))
    br = mathutils.Vector((-halfCubeLength, halfCubeLength, -halfCubeLength))
    bl = mathutils.Vector((halfCubeLength, halfCubeLength, -halfCubeLength))
    
    worldPoints = [tr,tl,br,bl]
    cameraPoints = []
    pixelPoints = []
    for p in worldPoints:
        cameraPoint = bpy_extras.object_utils.world_to_camera_view(scene,camera,p)
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

    data = bproc.renderer.render()

# for segmentation map
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance"]))


# write to file
    bproc.writer.write_hdf5("outputTest", data, append_to_existing_output=True)

# Write data to coco file
    bproc.writer.write_coco_annotations("outputTest",
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG",
                                    append_to_existing_output=True)



for i in range(1):
    generateImages()
    bproc.utility.reset_keyframes()

