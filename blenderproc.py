import blenderproc as bproc
import numpy as np
import bpy
import math


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



bproc.init()

cubeList = bproc.loader.load_blend("cube.blend")


objects = bpy.data.objects
faceCubes = []
for obj in objects:
    print(obj.name)
    if('FRONT' in obj.name):
        faceCubes.append(obj)

print(len(faceCubes))        

for cube in faceCubes:
    blue(cube)
# poi = bproc.object.compute_poi(c)

# cubes = bproc.object.get_all_mesh_objects()
# print(len(cubes))
# pretty sure their filter shit just doesn't work
# faceCubes = bproc.filter.by_attr(cubes,"name", "FRONT*", regex=True)
# faceCubes = []
# for cube in cubes:
#     if('FRONT' in cube.get_name()):
#         print(cube.get_name())
#         faceCubes.append(cube)

# for face in faceCubes:
#     white(face)

light = bproc.types.Light()
light.set_type("POINT")
light.set_location([3,-3,3])
light.set_energy(1000)


bproc.camera.set_resolution(512,512)


for i in range(5):
    # Sample random camera location above objects
    location = np.random.uniform([0, 1, 0], [0, .5, 0])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(np.array([0,0,0]) - location)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


bproc.renderer.render_segmap(map_by=['instance', 'class'])
bproc.renderer.set_noise_threshold(0.01)


data = bproc.renderer.render()

bproc.writer.write_hdf5("output", data)

