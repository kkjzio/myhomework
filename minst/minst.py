import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets



(x, y), (x_val, y_val) = datasets.mnist.load_data() 
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.   #  转换为浮点张量，并缩放到0~1
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转换为整形张量

y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) #构建数据集对象
train_dataset = train_dataset.batch(200)

 


model = keras.Sequential([   #  3 个非线性层的嵌套模型
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'), # 创建一层网络，设置输出节点数为 256，激活函数类型为 ReLU
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):

    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):  ## 迭代数据集对象，带 step 参数 或 for x,y in train_db:  迭代数据集对象



        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28*28)) #的参数−1表示当前轴上长度需要根据张量总元素不变的法则自动推导
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0] #  每个样本的平均误差


        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables) # 自动求导函数 tape.gradient(loss, model.trainable_variables)
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))# w' = w - lr * grad，更新网络参数

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())



def train():

    for epoch in range(30):

        train_epoch(epoch)






if __name__ == '__main__':
    train()