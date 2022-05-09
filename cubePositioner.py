import bpy
import cv2
import bpycv
import random
import numpy as np
import math
from time import sleep

faceCubes = [
    bpy.data.objects['single colored cube.001'],
    bpy.data.objects['single colored cube.003'],
    bpy.data.objects['single colored cube.004'],
    bpy.data.objects['single colored cube.005'],
    bpy.data.objects['single colored cube.006'],
    bpy.data.objects['single colored cube.007'],
    bpy.data.objects['single colored cube.008'],
    bpy.data.objects['single colored cube.009'],
    bpy.data.objects['single colored cube.010']]

#define rotation color mapping
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

for cube in faceCubes:
    yellow(cube)
    
# give each mesh uniq inst_id for segmentation
counter=1
for obj in bpy.data.objects:
    if obj.type == "MESH":
        obj = bpy.context.active_object
        obj["inst_id"] = counter
        counter=counter+1

# render image, instance annotation and depth in one line code
result = bpycv.render_data()

# write visualization inst|rgb|depth 
cv2.imwrite(
    "(inst|rgb|depth)." + str(n) + ".jpg", cv2.cvtColor(result.vis(), cv2.COLOR_RGB2BGR)
)