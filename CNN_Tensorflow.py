# -*- coding: utf-8 -*-
"""
@author: Chiu-Cheng-Chun

*ATTENTION:
1. You can use it at will, but please mark the source if you quote it for commercial use. Thanks~
2. If you have any questions feel free to contact me: craigchiu0619@gmail.com
"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig = plt.figure()

t0 = time.clock()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
定義訓練集
"""
X_Train = mnist.train.images#TrainingSet
X_Train = X_Train[0:8000,:]
Y_Train = mnist.train.labels#TrainingLabels
Y_Train = Y_Train[0:8000,:]

"""
定義變數
"""
W = []
b = []

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

"""
定義卷積
"""
def Convolution(inputs, num_of_filters, kernel_size):
    h = tf.layers.conv2d(inputs=inputs, filters=num_of_filters, kernel_size=kernel_size, padding="same", activation=tf.nn.relu)
    return h

"""
定義池化
"""
def Max_pool(conv):
    return  tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

"""
定義全聯接層模型
"""
def fc_model(X, n_neurons, W, b):
    for i in range(0, len(n_neurons) - 2):
        X = tf.nn.relu(tf.matmul(X,W[i]) + b[i])
    result = tf.matmul(X, W[-1]) + b[-1]
    return result

"""
定義損失函數，此為交叉熵代價函數
"""
def Cost(y_label, prediction):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_label))
    return cross_entropy


x_image = tf.reshape(x, [-1, 28, 28, 1])


epoches = 1500  #設定迭代次數
learning_rate = 0.01 #學習速率

"""
卷積格式:Convolution(輸入矩陣, filter個數, filter大小)
池化格式:Max_pool(輸入矩陣)
"""
h1 = Convolution(x_image, 5, [5, 5])#第一次卷積
p1 = Max_pool(h1)#第一次池化
h2 = Convolution(p1, 5, [5, 5])#第二次卷積
p2 = Max_pool(h2)#第二次池化

"""
將池化後的3維矩陣攤平成2維矩陣
"""
pool_times = 2
pool_flat = tf.reshape(p2, [-1, 7 * 7 * 5])
input_size = 7 * 7 * 5

n_neurons = [input_size, 100, 10]

"""
建構權重及偏差
"""
for i in range(0, len(n_neurons)-1):
    W.append(tf.Variable(tf.truncated_normal([n_neurons[i],n_neurons[i+1]])))
    b.append(tf.Variable(tf.zeros([1.0 ,n_neurons[i+1]])))


prediction = fc_model(pool_flat, n_neurons, W, b)
cross_entropy = Cost(y, prediction)
train = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#--------------------------------------------train-----------------------------------------

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

costs = np.ones((epoches+1,1))

"""
開始訓練
"""
for i in range(epoches+1):
    batch_xs, batch_ys = mnist.train.next_batch(150)
    sess.run(train, feed_dict = {x: batch_xs, y: batch_ys})
    costs[i] = sess.run(cross_entropy, feed_dict = {x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        print(i,"epoch(es)", " Done!")
        
    """
    顯示圖片及其辨識結果
    """
    if i % 200 == 0:
        res = sess.run(prediction, feed_dict = {x: batch_xs[0].reshape(1,784)})
        print("辨識結果為:", res.argmax())
        print("正確結果為:", batch_ys[0].argmax())
        one_pic_arr = np.reshape(batch_xs[0],(28,28)) 
        pic_matrix = np.matrix(one_pic_arr,dtype = "float")                     
        plt.imshow(pic_matrix)
        plt.show()

    
#plot
x_axis = np.arange(epoches)
fig, ax1 = plt.subplots()  
ax1.plot(x_axis, costs[1:], "green", label = "Cost")
ax1.set_ylabel('Cost值',fontproperties=myfont);
ax1.set_xlabel('訓練次數',fontproperties=myfont)
ax1.legend(loc=1,prop=myfont)
plt.show()

"""
計算訓練集準確度
"""
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Train Set Accuracy:", accuracy.eval(feed_dict={x: X_Train, y: Y_Train}) * 100, "%")

"""
開始測試
"""
X_Test = mnist.test.images
Y_Test = mnist.test.labels

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Set Accuracy:", accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}) * 100, "%")

    
print("總共費時:",time.clock()-t0,"秒")



