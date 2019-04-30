# -*- coding: utf-8 -*-
"""
@author: Chiu-Cheng-Chung

*ATTENTION:
1. You can use it at will, but please mark the source if you quote it for commercial use. Thanks~
2. If you have any questions feel free to contact me: craigchiu0619@gmail.com
"""

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig = plt.figure()

t0 = time.clock()#time start

#import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

#設置每層神經元數目
input_size = 784
output_size = 1024
n_neurons = [input_size, output_size]

#參數設置
x = tf.placeholder(tf.float32,[None, input_size])
W = tf.Variable(tf.truncated_normal([n_neurons[0],n_neurons[-1]]))
b_hidden = tf.Variable(tf.zeros([1.0 ,n_neurons[-1]]))
b_input = tf.Variable(tf.zeros([1.0 ,n_neurons[0]]))
learning_rate = 0.01
epoches = 2000
batch_size = 150
examples_to_show = 10 #結果要展示的圖片數量

#模型建構
def model():
    res = x
    z = tf.nn.sigmoid(tf.matmul(res, W) + b_hidden)
    res = tf.nn.sigmoid(tf.matmul(z, tf.transpose(W)) + b_input)
    return res

#定義損失函數，此為平方差代價函數
def Cost(x, prediction):
    square_error = tf.reduce_mean(tf.squared_difference(x, prediction))
    return square_error

out = model()
cost = Cost(x, out)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#---------------------------train-----------------------------------------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoches):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x: batch_xs})
    if i % 200 == 0:
        print("Epoch:", i,"cost=", sess.run(cost, feed_dict={x: batch_xs}))

#展示結果
rbm = sess.run(out, feed_dict={x: mnist.test.images[:examples_to_show]})
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(rbm[i], (28, 28)))
print("上圖為輸出結果，下圖為實際結果")
plt.show()

#計算總花費時間
print("總共費時:",time.clock()-t0,"秒")


