import numpy as np
import cv2 as cv
import csv
import random

# if __name__ == '__main__':
#     main(

colors = ['White', 'Yellow', 'Red', 'Orange', 'Green', 'Blue']  #
font = cv.FONT_HERSHEY_SIMPLEX         # 字体

# 读取数据
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    datas = [row for row in reader]
# print(datas)
random.shuffle(datas)
# 分组
random.shuffle(datas)
n = int(len(datas)/10)
test_set = datas[0:n]
train_set = datas[n:]



# 距离
def distance(d1, d2):
    res = 0
    for key in ('B', 'G', 'R'):
        # res += (int(d1[key]) - int(d2[key])) ** 2
        res += abs((int(d1[key]) - int(d2[key])))
    return res ** 0.5
                                
def knn(data, k=6):
    # 算距离
    res = [
        {'result':train['Color'], 'distance':distance(data, train)}
        for train in train_set
    ]
    res = sorted(res, key=lambda item:item['distance'])
    # print(res)
    res2 = res[0:k]
    # 加权平均
    result = {'White':0, 'Yellow':0, 'Red':0, 'Orange':0, 'Green':0, 'Blue':0}
    sum = 0
    for r in res2:
        sum += r['distance']
    for r in res2:
        result[r['result']] += 1-r['distance']/sum
    result2 = [result[key] for key in colors]
    # print(result2)
    res_color = colors[result2.index(max(result2))]
    # if res_color != res2[0]['result']:
    #     print('Error!')
    # return res_color
    return res2[0]['result']
    # print(res_color)
    # print(data['Color'])

# 鼠标回调函数， 鼠标双击图像，记录像素的HSV值
def Mousedouble(event, y, x, flag, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        # print(cube_img[x][y])
        point = np.uint8(cube_img[x][y])                      # 必要的
        # print(point)
        bgr_dict = {'B':point[0], 'G':point[1], 'R':point[2]}    # 将鼠标双击点的BGR值转为字典
        res_color = knn(bgr_dict, 6)                             # 是用KNN算计计算双击点颜色
        print(res_color)                                         # 输出识别出的颜色

# # 测试训练结果
# for i in range(10):
#     # 分组
#     random.shuffle(datas)
#     n = int(len(datas)/10)
#     test_set = datas[0:n]
#     train_set = datas[n:]
#     correct = 0
#     for test in test_set:
#         result = test['Color']
#         result2 = knn(test)
#         # print(result, result2)
#         if result == result2:
#             correct += 1
#         else:
#             print('T:',result, 'F:',result2)
#
#     print(str(correct)+' / '+str(len(test_set))+' = '+str(correct/len(test_set)))


# 开启摄像头，鼠标双击输出颜色
kernel = np.ones((5,5),np.float32)/25
cap = cv.VideoCapture(1)
cv.namedWindow('show_image')
cv.setMouseCallback('show_image', Mousedouble)
read_img = cv.imread('cube_1.png')
OpenVideo = 0    # 1:开启摄像头  0: 显示静态图
while(1):
    if OpenVideo:
        ret, frame = cap.read()
        frame = cv.GaussianBlur(frame, (9, 9), 0)   # 高斯模糊
        cube_img = cv.filter2D(frame,-1,kernel)
    else:
        cube_img = read_img
    show_img = cube_img.copy()  # 复制一张图像
    # cv.imshow('image', cube_img)
    cv.imshow('show_image', show_img)   # 显示图像
    key = cv.waitKey(20)
    if key & 0xFF == 27:      # 按下ESC键退出
        break