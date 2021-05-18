#---------------------------------------------#
#   该部分用于查看Xception网络结构
#---------------------------------------------#

from nets.deeplab import Xception
from keras.layers import Input

if __name__ == "__main__":
    input_shape=(512, 512, 3)
    alpha=1.
    OS=16
    img_input = Input(shape=input_shape)

    # x         64, 64, 2048
    # skip1     128, 128, 256
    model = Xception(img_input,alpha,OS=OS)

    
    model.summary()
    #for i in range(len(model.layers)):
    #    print(i,model.layers[i].name)
