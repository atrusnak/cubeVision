import cv2 as cv

cam_port=0
cam = cv.VideoCapture(cam_port)

result, image = cam.read()


if result: 
    imshow("GeeksForGeeks", image)

    imwrite("hello.png", image)

    waitKey(0)
    destroyWindow("GeeksForGeeks")


else:
    print("did'nt work")
