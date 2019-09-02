# -*- coding: utf-8 -*-
"""
@author: Chiu-Cheng-Chun

*ATTENTION:
1. You can use it at will, but please mark the source if you quote it for commercial use. Thanks~
2. If you have any questions feel free to contact me: craigchiu0619@gmail.com
"""

import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig = plt.figure()

#結構初始化
tf.reset_default_graph()

#開始計時
t0 = time.clock()

tf.set_random_seed(1)

#引入數據
mnist = input_data.read_data_sets('./input_data', one_hot=True, validation_size=10000)

#設定及初始化參數
learning_rate = 0.01
epoches = 5000
input_size = 28
n_steps = 28
batch_size = 150
output_size = 10
W = []
b = []

x = tf.placeholder(tf.float32, [None, n_steps, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

#設定各層神經元數量
n_hidden_units = 100
n_neurons = [input_size, n_hidden_units, output_size]

#建構權重及偏差
for i in range(0, len(n_neurons)-1):
    W.append(tf.Variable(tf.truncated_normal([n_neurons[i],n_neurons[i+1]])))
    b.append(tf.Variable(tf.zeros([1.0 ,n_neurons[i+1]])))

#RNN 模型建構
def RNN(X, W, b):
    #reshape X to 2-dimensional to compute
    X = tf.reshape(X, [-1, input_size])
    X_in = tf.matmul(X, W[0]) + b[0]
    
    #reshape X to 3-dimensional to put in dynamic_rnn
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
    #creat a lstm cell
    cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
    #tf.nn.dynamic_rnn expects a 3-dimensional tensor as input
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], W[-1]) + b[-1]  
    
    return results

#定義損失函數，此為平方差代價函數
def Cost(y_label, prediction):
    square_error = tf.reduce_mean(tf.squared_difference(y_label, prediction))
    return square_error

prediction = RNN(x, W, b)

cost = Cost(y, prediction)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

##-----------------------------train---------------------

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

costs = np.ones((epoches,1))

for i in range(epoches):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, input_size])
    batch_ys =batch_ys.reshape([batch_size, 10])
    sess.run(train, feed_dict={x: batch_xs,y: batch_ys}) 
    costs[i] = sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys})
    if i == 0 or i % 200 == 0 or i == epoches-1:
        print(i,"epoch(es)", " Done!")
##----------------------------plot------------------------

#plot
x_axis = np.arange(epoches-1)
fig, ax1 = plt.subplots()  
ax1.plot(x_axis, costs[1:], "green", label = "Cost")
ax1.set_ylabel('Cost值',fontproperties=myfont);
ax1.set_xlabel('訓練次數',fontproperties=myfont)
ax1.legend(loc=1,prop=myfont)
plt.show()

##----------------------------test------------------------
"""
計算訓練集準確度
"""
#定義訓練集
X_Train = mnist.train.images#TrainingSet
X_Train = X_Train[0:150,:]
Y_Train = mnist.train.labels#TrainingLabels
Y_Train = Y_Train[0:150,:]

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
X_Train = X_Train.reshape([-1, 28, 28])
Y_Train = Y_Train.reshape([-1, 10])
print("Train Set Accuracy:", sess.run(accuracy, feed_dict={x: X_Train, y: Y_Train,}) * 100, "%")

"""
開始測試
"""
X_Test = mnist.test.images
X_Test = X_Test[0:150,:]
Y_Test = mnist.test.labels
Y_Test = Y_Test[0:150,:]

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
X_Test = X_Test.reshape([-1, 28, 28])
Y_Test = Y_Test.reshape([-1, 10])
print("Test Set Accuracy:", sess.run(accuracy, feed_dict={x: X_Test, y: Y_Test,}) * 100, "%")

#結算總花費時間
print("總共費時:",time.clock()-t0,"秒")
