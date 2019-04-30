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
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig = plt.figure()

t0 = time.clock()

mnist = input_data.read_data_sets('./input_data', one_hot=True, validation_size=10000)

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
input_size = 784
output_size = 10

W = []
b = []

x = tf.placeholder(tf.float32,[None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

"""
定義每層神經元個數
"""
n_neurons = [input_size, 50, 50, output_size]

"""
建構權重及偏差
"""
for i in range(0, len(n_neurons)-1):
    W.append(tf.Variable(tf.truncated_normal([n_neurons[i],n_neurons[i+1]])))
    b.append(tf.Variable(tf.zeros([1.0 ,n_neurons[i+1]])))

"""
建構神經網路模型
"""
def model():
    res = x
    for i in range(0, len(n_neurons) - 2):
        res = tf.nn.relu(tf.matmul(res,W[i]) + b[i])
    res = tf.matmul(res, W[-1]) + b[-1]
    return res

"""
定義損失函數，Cost1為平方差代價函數，Cost2為交叉熵代價函數
"""
def Cost1(y_label, prediction):
    square_error = tf.reduce_mean(tf.squared_difference(y_label, prediction))
    return square_error

def Cost2(y_label, prediction):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_label))
    return cross_entropy


out = model()
cost = Cost2(y, out)#使用Cost2作為代價函數
learning_rate = 0.01
epoches = 2000

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)#定義學習演算法
#---------------------------

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

costs = np.ones((epoches+1,1))

for i in range(epoches+1):
    batch_xs, batch_ys = mnist.train.next_batch(500)
    sess.run(train_step, feed_dict = {x: batch_xs, y: batch_ys})
    costs[i] = sess.run(cost, feed_dict = {x: X_Train, y: Y_Train})
    if i % 100 == 0:
        print(i,"epoch(es)", " Done!")

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
correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Train Set Accuracy:", accuracy.eval(feed_dict={x: X_Train, y: Y_Train}) * 100, "%")

"""
開始測試
"""
X_Test = mnist.test.images
Y_Test = mnist.test.labels

correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(Y_Test,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Set Accuracy:", accuracy.eval(feed_dict={x: X_Test, y: Y_Test}) * 100, "%")

"""
計算所花時間
"""
print("總共費時:",time.clock()-t0,"秒")