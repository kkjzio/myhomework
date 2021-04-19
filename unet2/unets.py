# conding:utf-8

# By tf1！！！！！！！！！！！！！！！！
from __future__ import print_function
import tensorflow as tf
import numpy as np

def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def get_w_and_b(kernel_size, input_feature_size, output_feature_size, name):
    w = weight_variable([kernel_size, kernel_size, input_feature_size, output_feature_size], name=name + '_w')
    b = bias_variable([output_feature_size], name=name + '_b')
    return w, b

def get_deconv_w_and_b(kernel_size, input_feature_size, output_feature_size, name):
    w = weight_variable([kernel_size, kernel_size, input_feature_size, output_feature_size], name=name + '_w')
    b = bias_variable([input_feature_size], name=name + '_b')
    return w, b

def conv2d(x, W, b):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return conv_2d_b

def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def deconv2d(x, W, stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID', name="conv2d_transpose")

def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

batch_size=1
# 16x+124
image_w=572
image_h=572
channel=3
pool_size=2
nclass=2
down_layer = {}


x = tf.placeholder(tf.float32, shape=(batch_size, image_w, image_h, channel))
print ('input x:\t\t\t', x)
# layer 0:
w0, b0 = get_w_and_b(3, channel, 64, 'layer0')
x = conv2d(x, w0, b0)
print ('layer 0 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 64, 64, 'layer0_1')
x = conv2d(x, w01, b01)
print ('layer 0 conv2 output:\t\t', x)
down_layer['layer0'] = x
# layer 1:
print ('='*100)
x = max_pool(x, pool_size)
print ('layer 1 pool output:\t\t', x)
w0, b0 = get_w_and_b(3, 64, 128, 'layer1')
x = conv2d(x, w0, b0)
print ('layer 1 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 128, 128, 'layer1_1')
x = conv2d(x, w01, b01)
print ('layer 1 conv2 output:\t\t', x)
down_layer['layer1'] = x
# layer 2:
print ('='*100)
x = max_pool(x, pool_size)
print ('layer 2 pool output:\t\t', x)
w0, b0 = get_w_and_b(3, 128, 256, 'layer2')
x = conv2d(x, w0, b0)
print ('layer 2 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 256, 256, 'layer2_1')
x = conv2d(x, w01, b01)
print ('layer 2 conv2 output:\t\t', x)
down_layer['layer2'] = x
# layer 3:
print ('='*100)
x = max_pool(x, pool_size)
print ('layer 3 pool output:\t\t', x)
w0, b0 = get_w_and_b(3, 256, 512, 'layer3')
x = conv2d(x, w0, b0)
print ('layer 3 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 512, 512, 'layer3_1')
x = conv2d(x, w01, b01)
print ('layer 3 conv2 output:\t\t', x)
down_layer['layer3'] = x
# layer 4:
print ('='*100)
x = max_pool(x, pool_size)
print ('layer 4 pool output:\t\t', x)
w0, b0 = get_w_and_b(3, 512, 1024, 'layer4')
x = conv2d(x, w0, b0)
print ('layer 4 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 1024, 1024, 'layer4_1')
x = conv2d(x, w01, b01)
print ('layer 4 conv2 output:\t\t', x)
down_layer['layer4'] = x
# up layer 3:
print ('='*100)
w0, b0 = get_deconv_w_and_b(pool_size, 512, 1024, 'uplayer_3')
x = deconv2d(x, w0, pool_size) + b0
print ('uplayer 3 deconv2d output:\t', x)
x = crop_and_concat(down_layer['layer3'], x)
print ('uplayer 3 crop&concat output:\t', x)
w0, b0 = get_w_and_b(3, 1024, 512, 'uplayer3')
x = conv2d(x, w0, b0)
print ('uplayer 3 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 512, 512, 'uplayer3_1')
x = conv2d(x, w01, b01)
print ('uplayer 3 conv2 output:\t\t', x)
# up layer 2:
print ('='*100)
w0, b0 = get_deconv_w_and_b(pool_size, 256, 512, 'uplayer_2')
x = deconv2d(x, w0, pool_size) + b0
print ('uplayer 2 deconv2d output:\t', x)
x = crop_and_concat(down_layer['layer2'], x)
print ('uplayer 2 crop&concat output:\t', x)
w0, b0 = get_w_and_b(3, 512, 256, 'uplayer2')
x = conv2d(x, w0, b0)
print ('uplayer 2 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 256, 256, 'uplayer2_1')
x = conv2d(x, w01, b01)
print ('uplayer 2 conv2 output:\t\t', x)
# up layer 1:
print ('='*100)
w0, b0 = get_deconv_w_and_b(pool_size, 128, 256, 'uplayer_1')
x = deconv2d(x, w0, pool_size) + b0
print ('uplayer 1 deconv2d output:\t', x)
x = crop_and_concat(down_layer['layer1'], x)
print ('uplayer 1 crop&concat output:\t', x)
w0, b0 = get_w_and_b(3, 256, 128, 'uplayer1')
x = conv2d(x, w0, b0)
print ('uplayer 1 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 128, 128, 'uplayer1_1')
x = conv2d(x, w01, b01)
print ('uplayer 1 conv2 output:\t\t', x)
# up layer 0:
print ('='*100)
w0, b0 = get_deconv_w_and_b(pool_size, 64, 128, 'uplayer_0')
x = deconv2d(x, w0, pool_size) + b0
print ('uplayer 0 deconv2d output:\t', x)
x = crop_and_concat(down_layer['layer0'], x)
print ('uplayer 0 crop&concat output:\t', x)
w0, b0 = get_w_and_b(3, 128, 64, 'uplayer0')
x = conv2d(x, w0, b0)
print ('uplayer 0 conv1 output:\t\t', x)
w01,b01 = get_w_and_b(3, 64, 64, 'uplayer0_1')
x = conv2d(x, w01, b01)
print ('uplayer 0 conv2 output:\t\t', x)
# output layer
print ('='*100)
w0,b0 = get_w_and_b(1, 64, nclass, 'output_layer')
x = conv2d(x, w0, b0)
print ('output layer out:\t\t', x)
