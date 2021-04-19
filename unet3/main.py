import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from model import *
import matplotlib.pyplot as plt

from myGenerator import load_train, load_test


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


model = unet()
# 这里调用load_train(csvDir, width, height, batch_size)产生数据
# 如果内存小batch_size就设为1吧
# steps_per_epoch=2为load_train返回的次数，到所标记的次数算作完成一次epoch
# workers为最大进程数
history = model.fit_generator(load_train(r"object.csv", 512, 512, 2), workers=1,
                              steps_per_epoch=2, epochs=20,
                              validation_data=load_test(r"vale.csv", 512, 512, 100)
                              )


model.save('modelWithWeight.h5')
model.save_weights('fine_tune_model_weight')

# print(history.history)

# 展示一下精确度的随训练的变化图
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 展示一下loss随训练的变化图
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
