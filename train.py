# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#creat data
x_data = np.random.rand(100).astype(np.float32)     ##����ֵ[0��1)֮��������
y_data = x_data * 0.1 + 0.3     ##Ԥ��ֵ
###creat tensorflow structure strat###
# ����Ҫ��ϵ�����ģ��
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))    
biases = tf.Variable(tf.zeros([1]))     
y = Weights * x_data + biases   
# ������ʧ������ѵ������
loss = tf.reduce_mean(tf.square(y-y_data))   ##��С������    
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# ��ʼ������
init = tf.initialize_all_variables()
###creat tensorflow structure end###
# ����
sess = tf.Session()
sess.run(init)
# ѵ����ϣ�ÿһ��ѵ����Weights��biases���и���
for step in range(201):
    sess.run(train)
    if step % 20 == 0:  
            print(step,sess.run(Weights),sess.run(biases))  ##ÿ20�����һ��W��b