from time import sleep
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (1500,1500)
camera.start_preview()
for i in range(10):
    sleep(1)
    camera.capture('images/solved/solved' + str(i) + '.jpg')
camera.stop_preview()
