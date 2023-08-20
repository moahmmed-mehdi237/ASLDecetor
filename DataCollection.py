import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time



# for making the data files cuz i didnt wanna make them by hand

# parent_dir = 'E:\ASLDecetor\data\\'
# for letter in range(ord('N'),ord('Z')+1):
#     dir = chr(letter)
#     path = os.path.join(parent_dir,dir)
#     os.mkdir(path)
# os.mkdir(f'{parent_dir}space')

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 350
counter = 0

folder = 'E:\ASLDecetor\data\space'

while True:
    success , img = cap.read()
    hands , img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h  = hand['bbox']

        imgWhite = np.ones((imgsize,imgsize,3) , np.uint8)*255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        # imgWhite[:imgCropShape[0],:imgCropShape[1]] = imgCrop


        if h > w :
            k = imgsize/h
            w_cal = round(k * w)
            img_resize = cv2.resize(imgCrop , (w_cal,imgsize))
            img_resize_shape = img_resize.shape
            w_gap = round((imgsize - w_cal) / 2)
            imgWhite[: , w_gap:w_cal + w_gap] = img_resize


        else:
            k = imgsize/w
            h_cal = round(k*h)
            img_resize = cv2.resize(imgCrop , (imgsize , h_cal))
            img_resize_shape = img_resize.shape
            h_gap = round((imgsize-h_cal)/2)
            imgWhite[h_gap:h_cal+h_gap , :] = img_resize






        cv2.imshow('ImageCrop' , imgCrop)
        cv2.imshow('ImageWhite' , imgWhite)


    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == ord('l'):
        counter +=1
        print(counter)
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg' , imgWhite)