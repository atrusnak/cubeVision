from time import sleep
from picamera import PiCamera

camera = PiCamera()
camera.resolution = (1500,1500)
camera.start_preview()
sleep(5)
camera.capture('test.jpg')
camera.stop_preview()
