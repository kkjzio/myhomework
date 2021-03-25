import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

# 仅在需要时申请显存
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

'''
这个数据集x为28*28的灰度值图（0-255）
y为此图对应数值
'''
(x, y), (x_val, y_val) = datasets.mnist.load_data() 
# x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.   #  转换为浮点张量，并缩放到0~1
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转换为整形张量

# y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) #构建数据集对象
train_dataset = train_dataset.batch(200)

 
from tensorflow.keras import Sequential

network = Sequential([ # 网络容器LeNet-5
    layers.Conv2D(6,kernel_size=3,strides=1), # 第一个卷积层, 6 个 3x3 卷积核    3*3*6+6 （9权值对应1个偏值） 输出6*28*28
    layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层 
    layers.ReLU(), # 激活函数
    layers.Conv2D(16,kernel_size=3,strides=1), # 第二个卷积层, 16 个 3x3 卷积核   6*3*3*16+16
    layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
    layers.ReLU(), # 激活函数
    layers.Flatten(), # 打平层，方便全连接层处理
    layers.Dense(120, activation='relu'), # 全连接层，120 个节点
    layers.Dense(84, activation='relu'), # 全连接层，84 节点
    layers.Dense(10) # 全连接层，10 个节点
                    ])

# build 一次网络模型，给输入 X 的形状，其中 4 为随意给的 batchsz
network.build(input_shape=(4, 28, 28, 1))
# 统计网络信息
network.summary()


# 导入误差计算，优化器模块
from tensorflow.keras import losses, optimizers

# 创建损失函数的类，在实际计算时直接调用类实例即可
criteon = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(learning_rate=0.001)

# 记录预测正确的数量，总样本数量
correct, total = 0,0


def train_epoch(epoch):
    #声明全局变量
    global correct,total
    for x,y in train_dataset: # 遍历所有训练集样本


        # 构建梯度记录环境
        with tf.GradientTape() as tape:
            # 插入通道维度，=>[b,28,28,1]
            x = tf.expand_dims(x,axis=3)
            # 前向计算，获得 10 类别的概率分布，[b, 784] => [b, 10]
            out = network(x)
            # 真实标签 one-hot 编码，[b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # 计算交叉熵损失函数，标量
            loss = criteon(y_onehot, out)
            # 自动计算梯度
            grads = tape.gradient(loss, network.trainable_variables)
            # 自动更新参数
            optimizer.apply_gradients(zip(grads, network.trainable_variables))


        
        # 前向计算，获得 10 类别的预测分布，[b, 784] => [b, 10]
        out = network(x)
        # 真实的流程时先经过 softmax，再 argmax
        # argmax取最大数的索引（即变回y的数据格式）
        # 但是由于 softmax 不改变元素的大小相对关系，故省去
        pred = tf.argmax(out, axis=-1)
        y = tf.cast(y, tf.int64)
        # 统计预测正确数量
        correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y),tf.float32)))
        # 统计预测样本总数
        total += x.shape[0]


def train():

    for epoch in range(30):

        global correct,total

        train_epoch(epoch)
        
        # 计算准确率
        print('epoch:',epoch+1,'--test acc:', correct/total)

        correct, total = 0,0




if __name__ == '__main__':
    train()