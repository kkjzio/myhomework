import random
from tensorflow_core.python.keras.models import load_model
import numpy as np
import cv2

# # 加载模型h5文件
from MatteMatting import MatteMatting

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class ArrayEmpty(Exception):
    def __str__(self):
        return "预测列表为空，请使用Predict.add()往列表添加地址"


class Predict():
    def __init__(self, model_path, show_summary=False):
        self.item_list = []
        self.model = load_model(model_path)
        if show_summary:
            self.model.summary()

    def add(self, path):
        """
        :param path: 预测图片列表地址
        """
        self.item_list.append(path)

    def predict_all(self, model_in_size, original_size):
        """
        预测一组数据,并返回值
        :param model_out_size: 模型的输入尺寸(width,height)
        :param original_size: 图片原始尺寸。程序会自动将尺寸还原为这个尺寸(width,height)
        :return:迭代器返回生成结果
        """
        if len(self.item_list):
            for item in self.item_list:
                dc = self.predict_one(item, model_in_size, original_size)
                yield dc
        else:
            raise ArrayEmpty()

    def predict_one(self, path, model_in_size, original_size):
        """
        预测一个数据,并返回值
        :param path: 需要预测的数据
        :param model_out_size: 模型的输入尺寸(width,height)
        :param original_size: 图片原始尺寸。程序会自动将尺寸还原为这个尺寸(width,height)
        :return:
        """
        src = [path]
        get = self.__read_file(model_in_size, src=src)
        predict = self.model.predict(get)
        ii = 0
        dc = cv2.resize(predict[ii, :, :, :], original_size)  # 后面这个参数是形状恢复为原来的形状
        return dc

    @staticmethod
    def __read_file(size_tuple, src=[]):
        """
        规范化图片大小和像素值
        :param size_tuple: 图片大小，要求为元组(width,height)
        :param src:连接列表
        :return:返回预测图片列表
        """
        pre_x = []
        for s in src:
            print(s)
            input = cv2.imread(s)
            input = cv2.resize(input, size_tuple)
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            pre_x.append(input)  # input一张图片
        pre_x = np.array(pre_x) / 255.0
        return pre_x


if __name__ == '__main__':
    import os
    
    data_path_array = []
    # 需要预测的图片的文件夹
    resDir = r'E:\BaiduNetdiskDownload\carvana\train_hq2'
    # 将需要预测的图片地址存入数组
    for root, dirs, files in os.walk(resDir):
        for file in files:
            data_path_array.append(os.path.join(root, file))
    # print(data_path_array)
	# 这里使用我整理的那个类，实例化的时候先把训练得到的模型权重文件放进去
    pd = Predict("modelWithWeight.h5")
    # 然后调用我整理的那个类类里面的add方法把需要预测的地址添加进去
    for item in data_path_array:
        pd.add(item)
    # 调用里面的predict_all方法，返回的是一个生成器，需要我们用next来读取
    dd = pd.predict_all((512, 512), (1918, 1280))
    # 保存到指定位置
    for item in data_path_array:
        dc = next(dd)
        dc = (dc * 255).astype(np.uint8)  # 把dtype从float32转化为uint8
        item = cv2.imread(item)
        # dc.dtype='uint8'
        mm = MatteMatting(item, dc, input_type='cv2')
        mm.save_image(r"E:\BaiduNetdiskDownload\carvana\ppp\{}.png".format(str(random.randint(1, 10000000))), mask_flip=True)
