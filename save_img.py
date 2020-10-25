import numpy as np
import cv2 as cv
import kociemba
import matplotlib.pyplot as plt

# 读取六个面图像,200*200
cube = []
for i in range(6):
    cube.append(cv.imread('cube_'+str(i)+'.png'))
    # cv.imshow(str(i), cube[i])

def colorMatch(img):
    # cv.inRange函数色域范围
    lower = np.array([[66,0,100], [33,25,1], [138,65,71], [1,90,77], [55,62,63], [103,145,75]])
    upper = np.array([[120,62,255], [68,255,255], [198,255,255], [34,255,255], [103,255,255], [155,255,255]])
    masks = np.zeros((6,200,200), np.uint8)
    mask_2x1 = np.zeros((3, 400, 200), np.uint8)
    mask_2x3 = np.zeros((400, 600), np.uint8)
    kernel_3 = np.ones((3,3),np.uint8)#3x3的卷积核
    kernel_10 = np.ones((10,10),np.uint8)#10x10的卷积核

    # 分别得到6张掩膜
    # w, h = cube[0].shape[::-1]
    erosions = []
    # reses = []
    # edges = []
    # points = []
    # mins = []
    # maxs = []
    pixels = np.zeros((6,9), np.uint32)
    for i in range(6):
        cube_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        masks[i] = cv.inRange(cube_hsv, lower[i], upper[i])
        masks[i] = cv.morphologyEx(masks[i], cv.MORPH_OPEN, kernel_10)  # 开运算分割色块
        # cv.imshow(str(i), masks[i])
        erosions.append(cv.erode(masks[i], kernel_10, iterations=1))
        cv.imshow(str(i), erosions[i])
        # reses.append(cv.bitwise_and(img, img, mask=erosions[i]))
        # cv.imshow(str(i), reses[i])
        # edges.append(cv.Canny(masks[i], 50, 80, apertureSize=5))
        # cv.imshow(str(i), edges[i])
        # points.append(cv.findNonZero(edges[i]))    # 返回非零像素位置列表

        # 统计像素值总数
        pixels[i][0] = erosions[i][0:67, 0:67].sum() / 255
        pixels[i][1] = erosions[i][0:67, 67:134].sum() / 255
        pixels[i][2] = erosions[i][:67, 134:].sum() / 255
        pixels[i][3] = erosions[i][67:134, :67].sum() / 255
        pixels[i][4] = erosions[i][67:134, 67:134].sum() / 255
        pixels[i][5] = erosions[i][67:134,134:].sum() / 255
        pixels[i][6] = erosions[i][134:, :67].sum() / 255
        pixels[i][7] = erosions[i][134:, 67:134].sum() / 255
        pixels[i][8] = erosions[i][134:,134:].sum() / 255
    # print(pixels)
    ave = (pixels.max() + pixels.min()) / 2

    color_str = ['U', 'D', 'F', 'B', 'L', 'R']
    get_str = ''
    for i in range(9):
        count = 0
        for num in pixels[:,i]:
            if num > ave:
                # print(count)
                # print(color_str[count])
                get_str += color_str[count]
            count += 1

    return get_str

color = ''
# for i in range(6):
#     color += colorMatch(cube[i])
# print(color)
# key = kociemba.solve('DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD')
# print(key)
print(colorMatch(cube[0]))
cv.imshow('cube_0',cube[0])
cv.waitKey(0)




