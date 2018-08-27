import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def load_data(file_path):
    '''导入数据
    input: file_path(string):评分数据文件路径
    output：data():评分矩阵
    '''
    data = pd.read_csv(file_path, index_col=0)
    data = data.fillna(0)
    return data.values

def MF(data, k, learning_rate, beta, training_epochs, display_step=10):
    '''利用梯度下降法对矩阵进行分解
    input: 
        data: 评分矩阵
        k(int): 分解矩阵的参数
        learning_rate(float): 学习率
        beta(float): 正则化参数
        training_epochs(int): 最大迭代次数
    output: 
        U, V: 分解后的矩阵
    '''
    m, n = np.shape(data)

    # 1.初始化输入R
    R = tf.placeholder(tf.float32, [m, n])

    # 2.初始化U和V
    U = tf.get_variable("U", [m, k], initializer=tf.random_normal_initializer(0, 0.1))
    V = tf.get_variable("V", [n, k], initializer=tf.random_normal_initializer(0, 0.1))

    # 3.构建模型
    pred = tf.matmul(U, tf.transpose(V))
    regU = layers.l2_regularizer(beta)(U)
    regV = layers.l2_regularizer(beta)(tf.transpose(V))
    cost = tf.reduce_mean(tf.square(tf.subtract(R, pred))) + regU + regV
    # cost = tf.reduce_mean(tf.square(tf.subtract(R, pred)))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # 4.进行训练
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0
            # total_batch = int(m / batch_size)
            # for i in range(total_batch):
            #     batch = tf.train.shuffle_batch([data], batch_size)
            sess.run(train_step, feed_dict={R: data})
        
            # 打印cost
            if (epoch + 1) % display_step == 0:
                avg_cost = sess.run(cost, feed_dict={R: data})
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
        saver.save(sess, "model/mf_t/t")
        print("Optimization Finished!")

if __name__ == "__main__":
    data = load_data("data/t_rate.csv")
    MF(data, 5, 0.0002, 0.02, 50000, 1000)
