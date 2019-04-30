

# -*- coding: utf-8 -*-
"""
@author: Chiu-Cheng-Chung

*ATTENTION:
1. You can use it at will, but please mark the source if you quote it for commercial use. Thanks~
2. If you have any questions feel free to contact me: craigchiu0619@gmail.com
"""

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.font_manager as fm

#讓matplotlib補上中文顯示
myfont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
fig = plt.figure()

t0 = time.clock()

mnist = input_data.read_data_sets('./input_data', one_hot=True, validation_size=10000)

#------------------------------------------------Import Data---------------------------------------------------
    

class ANN(object):
    """
    初始化變數
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.ones([1, y]) * 1 for y in sizes[1:]]
        self.Weights = [np.random.randn(x, y) for x, y in zip(self.sizes[0:-1], self.sizes[1:])]
        for i in range(0,self.num_layers-2):
            self.Weights[i] = self.Weights[i] * np.sqrt(2 / self.sizes[i])
        self.Kill_Matrix = [np.ones([x, y]) for x, y in zip(self.sizes[0:-1], self.sizes[1:])]
        self.VdW = [np.zeros([x, y]) for x, y in zip(self.sizes[0:-1], self.sizes[1:])]
        self.SdW = [np.zeros([x, y]) for x, y in zip(self.sizes[0:-1], self.sizes[1:])]
        self.V_correction = [np.zeros([x, y]) for x, y in zip(self.sizes[0:-1], self.sizes[1:])]
        self.S_correction = [np.zeros([x, y]) for x, y in zip(self.sizes[0:-1], self.sizes[1:])]
        self.Vdb = [np.zeros([1, y]) * 1 for y in sizes[1:]]
        self.Sdb = [np.zeros([1, y]) * 1 for y in sizes[1:]]
        self.Vdb_correction = [np.zeros([1, y]) * 1 for y in sizes[1:]]
        self.Sdb_correction = [np.zeros([1, y]) * 1 for y in sizes[1:]]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10**(-8)
        self.epoches = 1
    
    """
    定義激勵函數softmax、ReLU及其微分
    """
    def Softmax(self, z):
        return (np.exp(z).T / np.sum(np.exp(z), axis = 1)).T
    
    def dSoftmax(self, a):
        return a * (1-a)
    
    def ReLU(self, z): 
        z[z<0]=0 
        return z
    
    def dReLU(self, a):
        return a > 0
    
    """
    定義Dropout
    """
    def Dropout(self, probability, a):
        self.Kill_Matrix = np.random.binomial(1, probability, size=a[0].shape)
        a = (a * self.Kill_Matrix) / probability #keep same expected value
        return a  
    
    """
    前饋式運算
    """
    def Forward_Prop(self, Input_Matrix, options, mode):
        self.Input = Input_Matrix
        self.examples = np.shape(self.Input)[0] 
        self.options = options
        self.a = {}
        self.a[0] = Input_Matrix                 
        for i in range(0,self.num_layers-1): 
            if mode == 1:#activate dropout
                if i != 0:
                    self.a[i] = ANN.Dropout(0.5, self.a[i])
            z = np.dot(self.a[i], self.Weights[i])
            z = self.biases[i] + z         
            if self.options[i] == 1:
                self.a[i+1] = ANN.Softmax(z)
            if self.options[i] == 2:
                self.a[i+1] = ANN.ReLU(z)
        self.h = self.a[self.num_layers-1]
        return self.h   
        
    """
    損失函數，此為交叉熵代價函數
    """
    def Cost_Func(self, Output_Matrix):
        self.Outputs = Output_Matrix
        cost = -np.sum(self.Outputs * np.log(self.h + 10**(-10)) + (1-self.Outputs) * np.log(1-self.h)) / self.examples
        return cost
    
    """
    定義Adam算法(用於修正學習速率)
    """
    def Adam(self, dW, i):
        self.VdW[i] = self.beta1 * self.VdW[i] + (1 - self.beta1) * dW
        self.SdW[i] = self.beta2 * self.SdW[i] + (1 - self.beta2) * (dW)**2
        self.V_correction[i] = self.VdW[i] / (1 - (self.beta1)**self.epoches)
        self.S_correction[i] = self.SdW[i] / (1 - (self.beta2)**self.epoches)     
        dW = self.V_correction[i] / (np.sqrt(self.S_correction[i]) + self.epsilon)
        return dW
    
    def Adam_db(self, db, i):
        self.Vdb[i] = self.beta1 * self.Vdb[i] + (1 - self.beta1) * db
        self.Sdb[i] = self.beta2 * self.Sdb[i] + (1 - self.beta2) * (db)**2
        self.Vdb_correction[i] = self.Vdb[i] / (1 - (self.beta1)**self.epoches)
        self.Sdb_correction[i] = self.Sdb[i] / (1 - (self.beta2)**self.epoches)     
        db = self.Vdb_correction[i] / (np.sqrt(self.Sdb_correction[i]) + self.epsilon)
        return db
    
    """
    倒傳遞運算
    """    
    def Back_Prop(self, alpha): 
        #Compute Output Delta
        Output_Del = [0.0] * self.h  #Initialize Output Delta 
        Output_Del = self.h - self.Outputs
            
        #Compute Hidden Delta
        self.Hidden_Del = {}
        for a in range(self.num_layers-3, -1, -1):
            self.Hidden_Del[a] = [0.0] * self.a[a+1]#Initialize a Hidden layer's Delta
            try:
                self.Hidden_Del[a] = np.dot(self.Hidden_Del[a+1], self.Weights[a+1].T) * ANN.dReLU(self.a[a+1])
            except:
                self.Hidden_Del[a] = np.dot(Output_Del, self.Weights[a+1].T) * ANN.dReLU(self.a[a+1])
                        
        #Update weights 
        for i in range(self.num_layers-2, -1, -1):            
            try:
                self.dW = ANN.Adam(np.dot(self.a[i].T , self.Hidden_Del[i])  / self.examples, i)
                self.Weights[i] = self.Weights[i] - alpha * self.dW
            except:
                self.dW = ANN.Adam(np.dot(self.a[i].T , Output_Del) / self.examples, i)
                self.Weights[i] = self.Weights[i] - alpha * self.dW
                
        #Update biases
        for i in range(self.num_layers-2, -1, -1):           
            try:
                self.db = ANN.Adam_db(np.sum(self.Hidden_Del[i], axis=0) / self.examples, i)
                self.biases[i] = self.biases[i] - alpha * self.db
            except:
                self.db = ANN.Adam_db(np.sum(Output_Del, axis=0) / self.examples, i)
                self.biases[i] = self.biases[i] - alpha * self.db
        self.epoches += 1

"""
定義訓練集
"""
X_Train = mnist.train.images#TrainingSet
X_Train = X_Train[0:5000,:]
Y_Train = mnist.train.labels#TrainingLabels
Y_Train = Y_Train[0:5000,:]
sizes = [784,50,50,10]#NUM of Neural units in every layers

"""
在設定每層的激勵函數時1:Softmax，2:ReLU
設定Mode時，1代表啟用Dropout
格式:ANN.Forward_Prop(訓練數據X,選擇激勵函數,MODE)
"""

ANN = ANN(sizes)
epochs = 800
cost = np.ones((epochs,1))

"""
開始訓練
"""
for i in range(1,epochs):#iterations
    ANN.Forward_Prop(X_Train,[2,2,1],0)#Import training set，and decided what activate function to use, and decided mode
    cost[i] = ANN.Cost_Func(Y_Train)
    ANN.Back_Prop(0.001)#learning rate
    if i == 0 or i % 100 == 0 or i == epochs-1:
        print(i,"epoch(es)", " Done!")


epochs = np.arange(epochs-1).reshape(epochs-1,1)#use to plot

#plot
fig, ax1 = plt.subplots()  
ax1.plot(epochs, cost[1:], "green", label = "Cost")
ax1.set_ylabel('Cost值',fontproperties=myfont);
ax1.set_yticks(np.linspace(0, np.max(cost), 10))
ax1.set_xlabel('訓練次數',fontproperties=myfont)
ax1.legend(loc=1,prop=myfont)
plt.show()

#Test overfitting
results_ = ANN.Forward_Prop(X_Train,[2,2,1],0)

X_predict=results_.argmax(1)
Y_answer=Y_Train.argmax(1)

Acc=X_predict==Y_answer

print("Train Set Accuracy:",(np.sum(Acc,axis=0)/len(X_Train)) * 100, "%")

#starting test
X_Test = mnist.test.images
Y_Test = mnist.test.labels
results = ANN.Forward_Prop(X_Test,[2,2,1],0)

X_predict=results.argmax(1)
Y_answer=Y_Test.argmax(1)

Acc=X_predict==Y_answer

print("Test Set Accuracy:",(np.sum(Acc,axis=0)/len(X_Test)) * 100, "%")
print("總共費時:",time.clock()-t0, "秒")
