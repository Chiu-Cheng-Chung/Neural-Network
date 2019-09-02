# -*- coding: utf-8 -*-
"""
@author: Chiu-Cheng-Chun

*ATTENTION:
1. You can use it at will, but please mark the source if you quote it for commercial use. Thanks~
2. If you have any questions feel free to contact me: craigchiu0619@gmail.com
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

t0 = time.clock()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

"""
宣告參數
"""
learning_rate = 0.005
epoches = 3000
examples_to_show = 10
batch_size = 275

n_input = 784

X = tf.placeholder("float", [None, n_input])

input_size = 784
output_size = 784

#batch normalization
scale_en = []
shift_en = []
scale_de = []
shift_de = []

W_en = []
b_en = []
W_de = []
b_de = []

"""
定義編碼器及解碼器的神經元數目
"""
en_n_neurons = [input_size, 256, 128]
de_n_neurons = [128, 256, output_size]

"""
建構權重及偏差
歸一化參數建構
"""
for i in range(0, len(en_n_neurons)-1):
    W_en.append(tf.Variable(tf.random_normal([en_n_neurons[i],en_n_neurons[i+1]])))
    b_en.append(tf.Variable(tf.zeros([en_n_neurons[i+1]])))
    W_de.append(tf.Variable(tf.random_normal([de_n_neurons[i],de_n_neurons[i+1]])))
    b_de.append(tf.Variable(tf.zeros([de_n_neurons[i+1]])))
    scale_en.append(tf.Variable(tf.ones([en_n_neurons[i+1]])))
    shift_en.append(tf.Variable(tf.zeros([en_n_neurons[i+1]])))
    scale_de.append(tf.Variable(tf.ones([de_n_neurons[i+1]])))
    shift_de.append(tf.Variable(tf.zeros([de_n_neurons[i+1]])))

"""
定義Batch Normalization
"""
def Batch_norm_en(Wx_plus_b, i):
    fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0, 1])
    Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift_en[i], scale_en[i], 10**(-3))
    return Wx_plus_b

def Batch_norm_de(Wx_plus_b, i):
    fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0, 1])
    Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift_de[i], scale_de[i], 10**(-3))
    return Wx_plus_b

"""
定義編碼器
"""
def encoder_model(x):
    res = x
    for i in range(0, len(en_n_neurons)-1):
        Wx_plus_b = tf.matmul(res,W_en[i]) + b_en[i]
        Wx_plus_b = Batch_norm_en(Wx_plus_b, i)
        res = tf.nn.sigmoid(Wx_plus_b)
    return res

"""
定義解碼器
"""
def decoder_model(x):
    res = x
    for i in range(0, len(de_n_neurons)-1):
        Wx_plus_b = tf.matmul(res,W_de[i]) + b_de[i]
        Wx_plus_b = Batch_norm_de(Wx_plus_b, i)
        res = tf.nn.sigmoid(Wx_plus_b)
    return res

"""
定義損失函數，此為平方差代價函數
"""
def Cost(x, prediction):
    square_error = tf.reduce_mean(tf.squared_difference(x, prediction))
    return square_error

encoder_output = encoder_model(X)
decoder_output = decoder_model(encoder_output)

cost = Cost(X, decoder_output)

train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

###########################Train#######################################

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for i in range(epoches):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={X: batch_xs})
    if i % 500 == 0 or i % epoches == 0:
        print("Epoch:", i,"cost=", sess.run(cost, feed_dict={X: batch_xs}))

###########################Test#######################################
    
encode_decode = sess.run(decoder_output, feed_dict={X: mnist.test.images[:examples_to_show]})
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))#印出結果
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.show()
    
print("總共費時:", time.clock()-t0, "秒")




