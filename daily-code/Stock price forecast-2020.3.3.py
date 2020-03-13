import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
import time

f=open('J:/文档e盘/深度学习/10 股票价格预测/10 股票价格预测/data_stocks.csv')
data=pd.read_csv(f)

data.describe()

data.info()

data.head()

print(time.strftime('%Y-%m-%d',time.localtime(data['DATE'].max())),
      time.strftime('%Y-%m-%d',time.localtime(data['DATE'].min())))

plt.plot(data['SP500'])
#要预测的这一列的曲线

data.drop('DATE',axis=1,inplace=True)

data_train=data.iloc[:int(data.shape[0]*0.9),:]
data_test=data.iloc[int(data.shape[0]*0.9):,:]
print(data_train.shape,data_test.shape)

scaler=MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train=scaler.transform(data_train)
data_test=scaler.transform(data_test)

x_train = data_train[:,1:]
y_train = data_train[:,0]
x_test = data_test[:,1:]
y_test = data_test[:,0]

input_dim=x_train.shape[1]
output_dim=1
batch_size=128
epochs=10
hidden1=1024
hidden2=512
hidden3=256
hidden4=128

tf.reset_default_graph()

X=tf.placeholder(shape=[None,input_dim],dtype=tf.float32)
Y=tf.placeholder(shape=[None],dtype=tf.float32)

w1=tf.get_variable('w1',[input_dim,hidden1],initializer=tf.contrib.layers.
                   xavier_initializer(seed=1))
b1=tf.get_variable('b1',[hidden1],initializer=tf.zeros_initializer())

w2=tf.get_variable('w2',[hidden1,hidden2],initializer=tf.contrib.layers.
                   xavier_initializer(seed=1))
b2=tf.get_variable('b2',[hidden2],initializer=tf.zeros_initializer())

w3=tf.get_variable('w3',[hidden2,hidden3],initializer=tf.contrib.layers.
                   xavier_initializer(seed=1))
b3=tf.get_variable('b3',[hidden3],initializer=tf.zeros_initializer())

w4=tf.get_variable('w4',[hidden3,hidden4],initializer=tf.contrib.layers.
                   xavier_initializer(seed=1))
b4=tf.get_variable('b4',[hidden4],initializer=tf.zeros_initializer())

w5=tf.get_variable('W5',[hidden4,output_dim],initializer=tf.contrib.layers.
                     xavier_initializer(seed=1))
b5=tf.get_variable('b5',[output_dim],initializer=tf.zeros_initializer())

h1=tf.nn.relu(tf.add(tf.matmul(X,w1),b1))
h2=tf.nn.relu(tf.add(tf.matmul(h1,w2),b2))
h3=tf.nn.relu(tf.add(tf.matmul(h2,w3),b3))
h4=tf.nn.relu(tf.add(tf.matmul(h3,w4),b4))
output=tf.transpose(tf.add(tf.matmul(h4,w5),b5))

cost=tf.reduce_mean(tf.squared_difference(output,Y))
train=tf.train.AdadeltaOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epo in range(epochs):
        shuffle_indices=np.random.permutation(np.arange(y_train.shape[0]))
        x_train=x_train[shuffle_indices]
        y_train=y_train[shuffle_indices]
        for i in range(y_train.shape[0]//batch_size):
            start=i*batch_size
            batch_x=x_train[start:start+batch_size,:]
            batch_y=y_train[start:start+batch_size]
            sess.run(train,feed_dict={X:batch_x,Y:batch_y})
            
            if i%1000==0:
                print('MSE Train',sess.run(cost,feed_dict={X:x_train,Y:y_train}))
                print('MSE Test',sess.run(cost,feed_dict={X:x_test,Y:y_test}))
                y_pred=sess.run(output,feed_dict={X:x_test})
                y_pred=np.squeeze(y_pred)
                plt.plot(y_test,label='test')
                plt.plot(y_pred,label='pred')
                plt.title('Epoch'+str(epo)+',Batch'+str(i))
                plt.legend()
                plt.show()
