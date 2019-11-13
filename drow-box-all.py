import cv2
import time
# import ysystem
import matplotlib.pyplot as plt
import numpy as np
import os
import imghdr
#from skimage import io
#from PIL import Image

#Image.LOAD_TRUNCATED_IMAGES = False

import time

def drawbox(img):
    weightsPath='yolov3-voc.backup'# 模型权重文件
    configPath="yolov3-voc.cfg"# 模型配置文件
    labelsPath = "voc.names"# 模型类别标签文件
    #初始化一些参数
    LABELS = open(labelsPath).read().strip().split("\n")
    boxes = []
    confidences = []
    classIDs = []

    #加载 网络配置与训练的权重文件 构建网络
    net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
    #读入待检测的图像
    image = cv2.imread(img)
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
            classID = np.argmax(scores)#np.argmax反馈最大值的索引
            confidence = scores[classID]
            if confidence >0.5:#过滤掉那些置信度较小的检测结果
                box = detection[0:4] * np.array([W, H, W, H])
                #print(box)
                (centerX, centerY, width, height)= box.astype("int")
                # 边框的左上角
                #print(centerX,centerY)
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                a = int(centerX + (width / 2))
                b = int(centerY + (height / 2))
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if a > W:
                    a = W
                if b > H:
                    b = H
                # 更新检测出来的框
                print(x,y,a,b)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    idxs=cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)
    try:
        box_seq = idxs.flatten()#[ 2  9  7 10  6  5  4]
    except:
        print(0,0,0,0)
        return image , 0
    if len(idxs)>0:
        for seq in box_seq:
            (x, y) = (boxes[seq][0], boxes[seq][1])  # 框左上角
            (w, h) = (boxes[seq][2], boxes[seq][3])  # 框宽高
            if classIDs[seq]==0: #根据类别设定框的颜色
                color = [0,0,255]
            else:
                color = [0,255,0]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # 画框
            return image,1
            #text = "{}: {:.4f}".format(LABELS[classIDs[seq]], confidences[seq])
            #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # 写字

"""
硬币批量裁剪测试
"""
def imgcut_basedoncoin_test(img_dir, destination_dir, failed_dir):
    file_dir = os.listdir(img_dir)
    print("file_dir:", file_dir)

    # 遍历匹配原图
    count_all= 0 #所有图片
    count_succed = 0 #匹配正确
    count_failed = 0  #倾斜导致尺寸错误
    for image_filename in file_dir:
        count_all = count_all + 1
        # 合成文件路径
        image_path = os.path.join(img_dir, image_filename)
        failedimage_path = os.path.join(failed_dir, image_filename)
        print("------------------------------------------------------------------")
        print("image_path:", image_path)
        image_name, ext = os.path.splitext(image_filename)
        # print("image_name:", image_name)
        # image_shellsize = image_name.split('-')[0]
        # print("image_shellsize:", image_shellsize)

        # 读取文件
        imgcuted, size = drawbox(image_path)

        # 用于统计colorgap 的使用次数
        # colorgap_list_matchcount[size] = int(colorgap_list_matchcount[size] + 1)

        image_newfilename = image_name + str(size) + ext
        image_newpath = os.path.join(destination_dir, image_newfilename)
        # print("size:", size)
        if size != 0:
            try:
                cv2.imwrite(image_newpath, imgcuted)
                count_succed = count_succed + 1
            except:
                print("save error!")
                count_failed = count_failed + 1
        else:
            cv2.imwrite(failedimage_path, imgcuted)
            count_failed = count_failed + 1
            print("failed! filename: %s"%image_filename)
        print("total %d images, %d tested, %d failed, success rate=%.3f"%(len(file_dir), count_all , count_failed, count_succed/count_all))

    # print("colorgap_list_matchcount:", colorgap_list_matchcount)
    # arg = np.argsort(colorgap_list_matchcount)[::-1]
    # print("arg:", arg)
    # colorgap_list_np = np.array(colorgap_list)
    # gap = colorgap_list_np[arg]
    # print("gap:", gap)


if __name__ == "__main__":

    img_dir = "jpgimages"
    destination_dir = "result/1/s"
    failed_dir = "result/1/f"
    imgcut_basedoncoin_test(img_dir, destination_dir, failed_dir)