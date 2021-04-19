import csv
import json
import os
import numpy as np
import cv2
from matplotlib import pyplot

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


def load_train(csvDir, width, height, batch_size):
    fx = 0.0
    fy = 0.0
    # 处理列表得到数组
    images_path = []
    labels_path = []
    # 利用csv.reader读取csv文件，然后将返回的值转化为列表
    # 然后就可以得到x（训练集）、y（标签）的地址
    csvFile = open(csvDir, "r")
    reader = csv.reader(csvFile)
    content = list(reader)
    for item in content:
        images_path.append(item[0])
        labels_path.append(item[1])
    # 进入循环读取照片
    while True:
    	# 下面定义两个数组来装每个批次(batch_size)的数据
        image_data_array = []
        label_data_array = []
        # 随机选一组数据
        index_group = np.random.randint(0, len(images_path), batch_size)
        # print("batch_size:", str(index_group))
        for index in index_group:
            image = images_path[index]
            label = labels_path[index]

            image_data = cv_imread(image)
            # 这里需要resize一下图片的长宽，让长宽与模型接收的长宽一致  interpolation=cv2.INTER_CUBIC为双线性插值
            # fx，fy沿x轴，y轴的缩放系数
            image_data = cv2.resize(image_data, (width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            image_data = image_data.astype(np.float32)
            image_data = np.multiply(image_data, 1.0 / 255.0)
            image_data_array.append(image_data)

            label_data = cv_imread(label)
            # label_data = cv2.cvtColor(label_data, cv2.COLOR_GRAY2BGR) # 颜色转化
            label_data = cv2.resize(label_data, (width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            label_data = label_data.astype(np.float32)
            label_data = np.multiply(label_data, 1.0 / 255.0)
            label_data_array.append(label_data)

        image_data_r = np.array(image_data_array)
        label_data_r = np.array(label_data_array)

        yield image_data_r, label_data_r

def load_test(csvDir, width, height, batch_size):
    fx = 0.0
    fy = 0.0
    # 处理列表得到数组
    images_path = []
    labels_path = []
    csvFile = open(csvDir, "r")
    reader = csv.reader(csvFile)
    content = list(reader)
    for item in content:
        images_path.append(item[0])
        labels_path.append(item[1])
    # 进入循环读取照片

    # for image, label in zip(images_path, labels_path):
    image_data_array = []
    label_data_array = []
    index_group = np.random.randint(0, len(images_path), batch_size)
    # print("batch_size:", str(index_group))
    for index in index_group:
        image = images_path[index]
        label = labels_path[index]

        image_data = cv_imread(image)
        image_data = cv2.resize(image_data, (width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        image_data = image_data.astype(np.float32)
        image_data = np.multiply(image_data, 1.0 / 255.0)
        image_data_array.append(image_data)

        label_data = cv_imread(label)
        # label_data = cv2.cvtColor(label_data, cv2.COLOR_GRAY2BGR) # 颜色转化
        label_data = cv2.resize(label_data, (width, height), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        label_data = label_data.astype(np.float32)
        label_data = np.multiply(label_data, 1.0 / 255.0)
        label_data_array.append(label_data)

    image_data_r = np.array(image_data_array)
    label_data_r = np.array(label_data_array)

    return image_data_r, label_data_r
