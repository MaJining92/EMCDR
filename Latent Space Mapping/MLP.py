import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from LM import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def MLP(input_Vs, input_Vt, beta, learning_rate, training_epochs, display_step=100):
    '''多层感知机映射函数
    input: 
        input_Vs(ndarray): 源领域矩阵
        input_Vt(ndarray): 目标领域矩阵 
        beta(float): 正则化参数
        learning_rate(float): 学习率
        training_epochs(int): 最大迭代次数
    output: 
        U, V: 分解后的矩阵
    '''

    k, m = np.shape(input_Vs)

    # 1. 初始化参数
    w1 = tf.Variable(tf.truncated_normal([2 * k, k], stddev = 0.1), name="w1")
    b1 = tf.Variable(tf.zeros([2 * k, 1]), name="b1")

    w2 = tf.Variable(tf.zeros([k, 2 * k]), name="w2")
    b2 = tf.Variable(tf.zeros([k, 1]), name="b2")
    
    Vs = tf.placeholder(tf.float32,[k, m])
    Vt = tf.placeholder(tf.float32,[k, m])
    
    # 2. 构建模型
    hidden1 = tf.nn.tanh(tf.matmul(w1, Vs)+b1)

    reg_w1 = layers.l2_regularizer(beta)(w1)
    reg_w2 = layers.l2_regularizer(beta)(w2)

    pred = tf.nn.sigmoid(tf.matmul(w2, hidden1) + b2)
    cost = tf.reduce_mean(tf.square(Vt - pred)) + reg_w1 + reg_w2
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    
    # 3. 开始训练
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            sess.run(train_step, feed_dict={Vs: input_Vs, Vt: input_Vt})

            if (epoch + 1) % display_step == 0:
                avg_cost = sess.run(cost, feed_dict={Vs: input_Vs, Vt: input_Vt})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        
        # 打印变量
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable:", k)
            print("Shape: ", v.shape)
            print(v)
        
        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, "model/mlp/mlp")
        print("Optimization Finished!")
    
if __name__ == "__main__":
    Us, Vs = load_data("model/mf_s/s.meta", "model/mf_s")
    # print(np.shape(Vs))
    Ut, Vt = load_data("model/mf_t/t.meta", "model/mf_t")

    beta = 0.001
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 10
    MLP(Vs, Vt, beta, learning_rate, training_epochs, display_step)