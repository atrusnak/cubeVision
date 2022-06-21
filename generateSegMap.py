import blenderproc as bproc
import numpy as np
import bpy
import math
import random
import csv


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
    for cube in faceCubes:
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
    with open("outputStickerSeg/scrambles.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(scramble)
    scramble = []

    # can compute point of interest if you want to point the camera. Just used origin tho
    # poi = bproc.object.compute_poi(c)

    # collect objects as blenderproc objects
    bprocCubes = bproc.object.get_all_mesh_objects()

    # pretty sure their filter shit just doesn't work
    # faceCubes = bproc.filter.by_attr(cubes,"name", "FRONT*", regex=True)

    # get face cubes and assign them to a different category for image segmentation later
    bprocFaceCubes = []
    c = 1
    for i in range(len(bprocCubes)):
        cube = bprocCubes[i]
        if('FRONT' in cube.get_name()):
            bprocFaceCubes.append(cube)
            bprocCubes[i].set_cp("category_id",c)
            c += 1
        # elif('025' in cube.get_name()):
            # bprocCubes[i].set_cp("category_id",99)



    bproc.camera.set_resolution(128,128)



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
    # for i in range(1):
    # Sample random camera location above objects
    location = np.random.uniform([0.1, 0.5, 0.1], [-0.1, 0.2, -0.1])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(np.random.uniform([0.05, 0.05, 0.05], [-0.05, -0.05, -0.05]) - location)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


    bproc.renderer.set_noise_threshold(0.01)

# set random hdri background and lighting
    #haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven("haven")
    #bproc.world.set_world_background_hdr_img(haven_hdri_path)

    data = bproc.renderer.render()


# for segmentation map
    data.update(bproc.renderer.render_segmap(map_by=["class"]))


# write to file
    bproc.writer.write_hdf5("outputStickerSeg", data, append_to_existing_output=True)


for i in range(1000):
    generateImages()
    bproc.utility.reset_keyframes()

