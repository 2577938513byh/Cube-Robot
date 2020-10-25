import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

kernel = np.ones((3,3),np.uint8)
cap = cv.VideoCapture(1)
each_faces  = np.zeros((6, 200, 200, 3), np.uint8)  #魔方的六个面
str_facenames = ["white", "yellow", "red", "origin", "green", "blue"]
i = 0

# cv.inRange函数色域范围
lower = np.array([[66,0,100], [33,25,1], [138,65,71], [1,90,77], [55,62,63], [103,145,75]])
upper = np.array([[120,62,255], [68,255,255], [198,255,255], [34,255,255], [103,255,255], [155,255,255]])
masks = np.zeros((6,200,200), np.uint8)
mask_2x1 = np.zeros((3, 400, 200), np.uint8)
mask_2x3 = np.zeros((400, 600), np.uint8)
kernel_3 = np.ones((3,3),np.uint8)#3x3的卷积核
kernel_10 = np.ones((10,10),np.uint8)#10x10的卷积核
while True:
    ret, frame = cap.read()
    # frame = cv.flip(frame, 1)
    cv.rectangle(frame, (99,99), (300, 300), (0, 0, 255), 1) # 画一个矩形框
    cv.imshow('frame', frame)
    cube_bgr = frame[100:300, 100:300] # 矩形框内的图像

    #cv.imshow('cube_bgr', cube_bgr)
    #cube_gray = cv.cvtColor(cube_bgr, cv.COLOR_BGR2GRAY)
    #cv.imshow('cube_gray', cube_gray)
    cube_hsv = cv.cvtColor(cube_bgr, cv.COLOR_BGR2HSV)
    i = 0
    count = 0
    while i<6:
        # 分别得到6张掩膜
        masks[i] = cv.inRange(cube_hsv, lower[i], upper[i])
        i += 1

    # 将6张掩膜拼接成一张总掩膜
    mask_2x1[0] = cv.vconcat((masks[0], masks[1]))
    mask_2x1[1] = cv.vconcat((masks[2], masks[3]))
    mask_2x1[2] = cv.vconcat((masks[4], masks[5]))
    mask_2x3 = cv.hconcat(mask_2x1[:])
    # 生成一张与总掩膜相同大小的图像
    cube_bgr_2x1 = cv.vconcat((cube_bgr,cube_bgr))
    cube_bgr_2x3 = cv.hconcat((cube_bgr_2x1,cube_bgr_2x1,cube_bgr_2x1))
    #
    erosion = cv.erode(mask_2x3, kernel_10, iterations=1)
    #res = cv.bitwise_and(cube_bgr_2x3, cube_bgr_2x3, mask = erosion)
    open = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel_10)
    cv.imshow('erosion', erosion)
    #cv.imshow('mask_2x3', mask_2x3)
    #cv.imshow('res', res)
    cv.imshow('open', open)
    k = cv.waitKey(10)

    if k == ord('m'):
        cv.imwrite(str(count) + '.png', cube_bgr)
        count+=1


    #
