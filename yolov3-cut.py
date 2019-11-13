import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

#获得硬币的中心位置，及可裁剪框的大小
def yolov3(img):
    weightsPath='yolov3-voc.backup'# 模型权重文件
    configPath="yolov3-voc.cfg"# 模型配置文件

    #加载 网络配置与训练的权重文件 构建网络
    net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
    #读入待检测的图像
    #image = cv2.imread(img)
    image = img.copy()
        #得到图像的高和宽
    (H,W) = image.shape[0:2]

    # 得到 YOLO需要的输出层
    ln = net.getLayerNames()
    out = net.getUnconnectedOutLayers()#得到未连接层得序号  [[200] /n [267]  /n [400] ]
    x = []
    for i in out:   # 1=[200]
        x.append(ln[i[0]-1])    # i[0]-1    取out中的数字  [200][0]=200  ln(199)= 'yolo_82'
    ln=x
    # ln  =  ['yolo_82', 'yolo_94', 'yolo_106']  得到 YOLO需要的输出层

    #从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
    #blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None, crop=None, ddepth=None)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)#构造了一个blob图像，对原图像进行了图像的归一化，缩放了尺寸 ，对应训练模型

    net.setInput(blob) #将blob设为输入？？？ 具体作用还不是很清楚
    layerOutputs = net.forward(ln)  #ln此时为输出层名称  ，向前传播  得到检测结果

    for output in layerOutputs:  #对三个输出层 循环
        for detection in output:  #对每个输出层中的每个检测框循环
            scores=detection[5:]  #detection=[x,y,h,w,c,class1,class2] scores取第6位至最后
            #print(scores)
            classID = np.argmax(scores)#np.argmax反馈最大值的索引
            confidence = scores[classID]
            if confidence >0.5:#过滤掉那些置信度较小的检测结果
                box = detection[0:4] * np.array([W, H, W, H])
                #print(box)
                (centerX, centerY, width, height)= box.astype("int")
                # 边框的左上角
                list1 = [centerX,centerY,width,height]
                if list1 is None:
                    return [0,0,0,0]
                # print(centerX,centerY)
                return centerX,centerY,width,height

#对预测的硬币位置进行裁剪
#输入圆心的位置，模型检测到的宽高以及原图
#输出更为精确的硬币剪裁图片
def yolov3crop(centerX, centerY, width, height, img):
    (H, W) = img.shape[0:2]
    x = int(centerX - (width / 2)) - 50
    a = int(centerY - (height / 2)) - 50
    y = int(centerX + (width / 2)) + 50
    b = int(centerY + (height / 2)) + 50
    if x < 0:
        x = 0
    if y > W:
        y = W
    if a < 0:
        a = 0
    if b > H:
        b = H
    #print(x, y, a, b)
    crop = img[a:b, x:y]
    return crop

#获得硬币的半径
def getcoinR(crop, colorgap):
    image = crop.copy()
    # 转换为灰度图
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 二值化
    ret, image = cv2.threshold(image, colorgap, 255, cv2.THRESH_BINARY)
    # 中值滤波
    image = cv2.medianBlur(image, 5)
    # 开运算(open)：先腐蚀后膨胀的过程。
    kernel = np.ones((50, 50), np.uint8)
    # 设置方框大小及类型
    # cv2.morphologyEx(src, type, kernel)
    # src 原图像 type 运算类型 kernel 结构元素
    # cv2.MORPH_OPEN 进行开运算，指的是先进行腐蚀操作，再进行膨胀操作
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # 闭运算(close)：先膨胀后腐蚀的过程。
    kernel = np.ones((50, 50), np.uint8)
    # cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # 轮廓提取,并且绘制所有轮廓
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 边缘过多，图片过于复杂，可能时非相关图片，直接返回
    if len(contours) > 20:
        # print("image too complicated, len(contours) = ", len(contours))
        return 0
    # 遍历所有轮廓,找出最大圆形的半径，圆形坐标
    circle_maxR = 0  # 最大圆半径
    for cnt in range(len(contours)):
        # 轮廓逼近，逼近后绘制轮廓
        epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
        approx = cv2.approxPolyDP(contours[cnt], epsilon, True)

        # 可能是圆形，进入
        if len(approx) >= 8:
            # 求面积和周长
            P = cv2.arcLength(contours[cnt], True)
            S = cv2.contourArea(contours[cnt])
            # 通过面积计算半径
            R_basedonS = int((S / 3.1415) ** 0.5)
            # print("R_basedonS=", R_basedonS)
            # 通过周长计算半径
            R_basedonP = int(P / 2 / 3.1415)
            # print("R_basedonP=", R_basedonP)
            # 你和R值，为两种计算方式的均值
            # R = int(((R_basedonP ** 2 + R_basedonS ** 2) / 2) ** 0.5)
            # 两种计算方式取较小值作为半径
            R = min(R_basedonS, R_basedonP)
            # print("R=", R)
            # 计算两种计算方式半径之间的比例关系
            if R_basedonS < R_basedonP and R_basedonS != 0:
                R_scale = (R_basedonP - R_basedonS) / R_basedonS
            elif R_basedonP < R_basedonS and R_basedonP != 0:
                R_scale = (R_basedonS - R_basedonP) / R_basedonP
            else:
                # print("contours[%d] result: not rect or circle! passed." % cnt)
                continue
            # print("R_scale=", R_scale)

            # 判断是否是真的圆
            # 上下级差大于4倍的半径。
            # 两种方式计算的圆的半径相差不大于20%
            if  R_scale < 0.3:
                # 存储最大圆半径
                if circle_maxR < R:
                    circle_maxR = R
                # print("contours[%d] result: p: %.3f, S: %.3f, tpye: %s, R：%.3f" % (cnt, P, S, "circle", R))
            #else:
                #print("contours[%d] result: not rect or circle! passed." % cnt)
        else:
            # print("contours[%d] result: not rect or circle! passed." % cnt)
            continue
    # print("circle_maxR=", circle_maxR)
    return circle_maxR

#输入圆心位置、半径、以及原图
#若可以剪裁，则输出剪裁得到的纹理及其尺寸
#若不能剪裁，则输出原图
def imgcut(circle_middle_maxX, circle_middle_maxY, circle_maxR, img):
    # 判断x方向最大可截取范围
    if circle_middle_maxX < img.shape[0] / 2:
        range_x_min = circle_middle_maxX + circle_maxR
        range_x_max = img.shape[0]
        # print("on the top side.")
    else:
        range_x_min = 0
        range_x_max = circle_middle_maxX - circle_maxR
        # print("on the down side.")
    hight_max = range_x_max - range_x_min
    # print("hight_max =", hight_max)

    # 判断y方向最大可截取范围
    if circle_middle_maxY < img.shape[1] / 2:
        range_y_min = circle_middle_maxY + circle_maxR
        range_y_max = img.shape[1]
        # print("on the left side.")
    else:
        range_y_min = 0
        range_y_max = circle_middle_maxY - circle_maxR
        # print("on the right side.")
    width_max = range_y_max - range_y_min
    #print("width_max =", width_max)

    # 判断长宽方向上总体最大可截取范围
    hight_and_width_min = min(width_max, hight_max)
    # print("hight_and_width_min =", hight_and_width_min)
    # 确定截取尺寸为2.5 5 10 15cm
    scale_cut = hight_and_width_min / circle_maxR
    #print("scale_cut =", scale_cut)
    if scale_cut >= 4 and scale_cut < 8:
        scale_of_R = 4
    # elif scale_cut >= 2 and scale_cut < 4:
    #     scale_of_R = 2
    elif scale_cut >= 8 and scale_cut < 12:
        scale_of_R = 8
    elif scale_cut >= 12:
        scale_of_R = 12
    else:
        scale_of_R = 0
        return img, 0
    #print("scale_of_R =", scale_of_R)
    # print("circle_maxR =", circle_maxR)

    # 计算x y方向上的截取范围
    width_y_min = 0
    width_y_max = 0
    hight_x_min = 0
    hight_x_max = 0
    # 计算x截取范围
    if circle_middle_maxX < img.shape[0] / 2:
        # hight_x_min = circle_middle_maxX + circle_maxR
        # hight_x_max = hight_x_min + scale_of_R * circle_maxR
        hight_x_max = img.shape[0]
        hight_x_min = hight_x_max - scale_of_R * circle_maxR
    else:
        # hight_x_max = circle_middle_maxX - circle_maxR
        # hight_x_min = hight_x_max - scale_of_R * circle_maxR
        hight_x_min = 0
        hight_x_max = scale_of_R * circle_maxR
    # print("hight_x_min =", hight_x_min)
    # print("hight_x_max =", hight_x_max)
    hight = hight_x_max - hight_x_min
    print("hight =", hight)

    # 计算y截取范围
    if circle_middle_maxY < img.shape[1] / 2:
        # width_y_min = circle_middle_maxY + circle_maxR
        # width_y_max = width_y_min + scale_of_R * circle_maxR
        width_y_max = img.shape[1]
        width_y_min = width_y_max - scale_of_R * circle_maxR
    else:
        # width_y_max = circle_middle_maxY - circle_maxR
        # width_y_min = width_y_max - scale_of_R * circle_maxR
        width_y_min = 0
        width_y_max = scale_of_R * circle_maxR
    # print("width_y_min =", width_y_min)
    # print("width_y_max =", width_y_max)
    width = width_y_max - width_y_min
    print("width =", width)

    image_cuted = img[hight_x_min:hight_x_max, width_y_min:width_y_max, :]
    # if log_output == 1:
    #     plt.figure('result')
    #     plt.imshow(image_cuted)

    # size = int(scale_of_R / 4 * 5)
    size = scale_of_R * 5 / 4
    # print("coin found，size =", size)
    # plt.show()
    return image_cuted, size

#对图像进行剪裁，输入为一张图片
# 若成功剪裁，则返回一张剪裁后的纹理图及其尺寸
# 若无法得到合适的剪裁区域，则返回原图和0
def imgcut_basedoncoin_yolov3(img):
    colorgap_list = [50, 20, 70, 130, 90, 35, 80, 65, 15, 75, 100, 55, 120, 30, 25, 105, 110, 185, 95]
    # colorgap_list = [130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
    #colorgap_list = [120]
    #colorgap_list = [100]
    centerX, centerY, width, height = yolov3(img)
    # print(centerX, centerY, width, height)
    crop = yolov3crop(centerX, centerY, width, height, img)
    cv2.imwrite("2.jpg",crop)
    for colorgap_index in range(len(colorgap_list)):
        r = getcoinR(crop, colorgap_list[colorgap_index])
        if r == 0:
            r = int((width + height)/4)
        print(r)
        image, size = imgcut(centerY, centerX, r, img)
        # print("coin NOT found, colorgap is %d"%colorgap_list[colorgap_index])
        # if image == img:
        #     image, size = crop_coin_above(image)
        if size != 0:
            # print("coin found, colorgap is %d"%colorgap_list[colorgap_index])
            # return image, colorgap_index + 1
            print("size =", size)
            return image, size
    return img , 0


img = cv2.imread("jpgimages/b0094f.jpg")
img,size = imgcut_basedoncoin_yolov3(img)
cv2.imwrite("11.jpg", img)