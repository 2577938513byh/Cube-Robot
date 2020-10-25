# 魔方颜色识别KNN算法
# HSV训练数据采集
# 双击图像记录
# 按下空格键记录九个方块数据
# 按ESC键退出并保存


import numpy as np
import cv2 as cv
import csv
import random
# 定义无意义的函数
def nothing(x):
    pass

def GetRandomPoints(midpoint, radius, num):
    '''
    函数功能： 获取中心点半径范围内的随机点
    参数：midpoints:中心点坐标
         radius：半径
         num：需要获取随机点的个数
    返回值：随机点坐标列表
    '''
    random_points = []
    for i in range(num):
        x = random.randint(midpoint[0]-radius, midpoint[0]+radius)
        y = random.randint(midpoint[1] - radius, midpoint[1] + radius)
        random_points.append((x, y))
    return random_points

def DrawSukudo(image, rect_pt1, rect_pt2, color):
    '''
    函数功能：在图像中画出九宫格
    参数：image:输入图像
         rect_pt1:左上角顶点
         rect_pt2:右下角顶点
         color：颜色BRG
    返回值:含有九宫格的图像
    '''
    cv.rectangle(image, rect_pt1, rect_pt2, color)
    cv.line(image, (rect_pt1[0] + int(width / 3), rect_pt1[1]), (rect_pt2[0] - int(2 * width / 3), rect_pt2[1]), color)
    cv.line(image, (rect_pt1[0] + int(2 * width / 3), rect_pt1[1]), (rect_pt2[0] - int(width / 3), rect_pt2[1]), color)
    cv.line(image, (rect_pt1[0], rect_pt1[1] + int(width / 3)), (rect_pt2[0], rect_pt2[1] - int(2 * width / 3)), color)
    cv.line(image, (rect_pt1[0], rect_pt1[1] + int(2 * width / 3)), (rect_pt2[0], rect_pt2[1] - int(width / 3)), color)
    return image

def GetBGR(image, point):
    '''
    函数功能：获取图像中像素点的BGR
    :param image: 图像
    :param point: 点坐标
    :return: 含有BGR的字典
    '''
    x = point[0]
    y = point[1]
    point = np.uint8([[cube_img[x][y]]])
    bgr_dict = {'B': point[0][0][0], 'G': point[0][0][1], 'R': point[0][0][2], 'Color': colors[color_index]}  # 生成字典
    return bgr_dict

random_points = []
width = 300  # 定义正方形大小
frame_shape = [640, 480]
rect_pt1 = (int((frame_shape[0]-width)/2), int((frame_shape[1]-width)/2))
rect_pt2 = (int(rect_pt1[0]+300), int(rect_pt1[1]+300))
step = int(width/6)
# 获取九个方块的中心点
midpoints = []
for i in range(3):
    for j in range(3):
        midpoints.append((rect_pt1[0] + (2*j+1)*step, rect_pt1[1] + (2*i+1)*step))
print(midpoints)

radiu = 20
kernel = np.ones((5,5),np.float32)/25
font = cv.FONT_HERSHEY_SIMPLEX         # 字体
headers = ['B', 'G', 'R', 'Color']     # CSVdeheader
colors = ['White', 'Yellow', 'Red', 'Orange', 'Green', 'Blue']  #
count = 0    # 记录采集数据数量
# rows = []  # 写入CSV数据
# f = open('train_sets.csv', 'w', newline='')
with open('test.csv', 'a', newline='')as f:  # 打开文件
    f_csv = csv.DictWriter(f, headers)            #
    f_csv.writeheader()                           # 写入header

    # 鼠标回调函数， 鼠标双击图像，记录像素的HSV值
    def draw_circle(event, y, x, flag, param):
        global count
        if event == cv.EVENT_LBUTTONDBLCLK:
            print(cube_img[x][y])
            point = np.uint8([[cube_img[x][y]]])              # 必要的
            # hsv = cv.cvtColor(point, cv.COLOR_BGR2HSV)     # 将bgr数据转为hsv
            hsv = point                                # 直接用BGR，不用HSV
            # print(hsv[0][0])
            bgr_dict = {'B':point[0][0][0], 'G':point[0][0][1], 'R':point[0][0][2], 'Color':colors[color_index]}   # 生成字典
            # rows.append(hsv_dict)
            count += 1
            print(count, bgr_dict, cube_img.dtype)          # 输出当前采集的的数据及数据个数
            f_csv.writerow(bgr_dict)        # 写入数据



    cap = cv.VideoCapture(1)

    # img = cv.imread('cube_0.png')
    cv.namedWindow('show_image')
    cv.setMouseCallback('show_image', draw_circle)
    cv.createTrackbar('Color', 'show_image', 0, 5, nothing)
    color_index = 0   # 颜色的索引
    while(1):
        ret, frame = cap.read()
        frame = cv.GaussianBlur(frame, (9, 9), 0)   # 高斯模糊
        cube_img = cv.filter2D(frame,-1,kernel)
        show_img = cube_img.copy()  # 复制一张图像，在复制出来的图像上画九宫格，不影响原图像
        cv.putText(show_img, colors[color_index], (0, 25), font, 1, (0, 0, 255), 2, cv.LINE_AA) # 显示当前采集颜色
        show_img = DrawSukudo(show_img, rect_pt1, rect_pt2, (0, 0, 255))
        for point in midpoints:
            # cv.circle(show_img, point, 25, (0, 0, 255))
            # 在九个方块中心点画正方形，确保正方形内的颜色一致
            cv.rectangle(show_img, (point[0]-radiu, point[1]-radiu), (point[0]+radiu, point[1]+radiu), (0, 0, 255))
        # cv.imshow('image', cube_img)
        cv.imshow('show_image', show_img)   # 显示含有九宫格的图像
        color_index = cv.getTrackbarPos('Color', 'show_image')   # 从轨迹栏获取当前采集颜色的索引
        key = cv.waitKey(20)
        if key & 0xFF == 27:      # 按下ESC键退出
            break
        if key & 0xFF == 32:      # 按下空格
            for point in midpoints:
                num = 10
                random_points = GetRandomPoints(point, 20, num)
                b, g, r = 0, 0, 0
                for rpoint in random_points:
                    # print(rpoint)
                    x, y = rpoint[:]
                    b += int(cube_img[x][y][0] / num)
                    g += int(cube_img[x][y][1] / num)
                    r += int(cube_img[x][y][2] / num)
                bgr_dict = {'B':b, 'G':g, 'R':r, 'Color':colors[color_index]}   # 生成字典
                count += 1
                print(count, bgr_dict, cube_img.dtype)  # 输出当前采集的的数据及数据个数
                f_csv.writerow(bgr_dict)  # 写入数据

    cv.destroyAllWindows()
