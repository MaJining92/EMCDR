import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def load_data(meta_path, ckpt_path):
    '''加载参数
    input: 
        meta_path(str): 图路径
        ckpt_path(str): 保存目录
    output: 
        U, V(ndarray): 潜因子
    '''

    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        loader = tf.train.import_meta_graph(meta_path)
        loader.restore(sess, tf.train.latest_checkpoint(ckpt_path))

        U, V = sess.run(["U:0", "V:0"])
    return U.T, V.T

def linear_mapping(input_Vs, input_Vt, beta, learning_rate, training_epochs, display_step=100):
    '''线性映射函数
    input: 
        input_Vs(ndarray): 源领域矩阵
        input_Vt(ndarray): 目标领域矩阵
        beta(float): 正则化参数
        learning_rate(float): 学习率
        training_epochs(int): 最大迭代次数
        display_step(int): 展示步数
    output: 
        M, b: 映射函数参数
    '''
    k, m = np.shape(input_Vs)

    # 1. 设置变量
    Vs = tf.placeholder(tf.float32, [k, m])
    Vt = tf.placeholder(tf.float32, [k, m])
    M = tf.get_variable("M", [k, k], initializer=tf.random_normal_initializer(0, 0.1))
    b = tf.Variable(tf.zeros([m]), name="b")

    # 2. 构造模型
    predVt = tf.matmul(M, Vs) + b
    regM = layers.l2_regularizer(beta)(M)
    cost = tf.reduce_mean(tf.square(Vt - predVt)) + regM
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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
        saver.save(sess, "model/lm/lm")
        print("Optimization Finished!")

if __name__ == "__main__":
    Us, Vs = load_data("model/mf_s/s.meta", "model/mf_s")
    # print(np.shape(Vs))
    Ut, Vt = load_data("model/mf_t/t.meta", "model/mf_t")

    beta = 0.001
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 10
    linear_mapping(Vs, Vt, beta, learning_rate, training_epochs, display_step)
