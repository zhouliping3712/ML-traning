# -*- coding: utf-8 -*-
import tensorflow as tf
import logging

# 设置日志
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# RNN类
class RNN:

    # 初始化
    # 参数说明： element_size:      元素大小
    #           time_steps:        序列大小
    #           num_classes:       目标变量的类别总数
    #           batch_size:        图片总数
    #           hidden_layer_size: 隐藏层的神经元个数
    #           epoch:             训练次数
    #           learning_rate:     用RMSProp优化时的学习率
    #           save_model_path:   模型保存地址
    def __init__(self, element_size, time_steps, num_classes, batch_size, hidden_layer_size = 150,
                 epoch = 1000, learning_rate=0.001, save_model_path = r'./logs/RNN_train.ckpt'):

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.save_model_path = save_model_path

        # 设置RNN结构
        self.element_size = element_size
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size

        # 输入向量和输出向量
        self._inputs = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.element_size], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='inputs')

        # 利用TensorFlow的内置函数BasicRNNCell, dynamic_rnn来构建RNN的基本模块
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_layer_size)
        outputs, _ = tf.nn.dynamic_rnn(rnn_cell, self._inputs, dtype=tf.float32)
        Wl = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.num_classes], mean=0, stddev=.01))
        bl = tf.Variable(tf.truncated_normal([self.num_classes], mean=0, stddev=.01))

        def get_linear_layer(vector):
            return tf.matmul(vector, Wl) + bl

        # 取输出的向量outputs中的最后一个向量最为最终输出
        last_rnn_output = outputs[:, -1, :]
        self.final_output = get_linear_layer(last_rnn_output)

        # 定义损失函数并用RMSProp优化
        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=self.final_output, labels=self.y)
        self.cross_entropy = tf.reduce_mean(softmax)
        self.train_model = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cross_entropy)

        self.saver = tf.train.Saver()
        logger.info('Initialize RNN model...')

    # 模型训练
    def train(self, x_data, y_data):

        logger.info('Training RNN model...')
        with tf.Session() as sess:
            # 对所有变量进行初始化
            sess.run(tf.global_variables_initializer())

            # 进行迭代学习
            feed_dict = {self._inputs: x_data, self.y: y_data}
            for i in range(self.epoch + 1):
                sess.run(self.train_model, feed_dict=feed_dict)
                if i % int(self.epoch / 50) == 0:
                    # to see the step improvement
                    print('已训练%d次, loss: %s.' % (i, sess.run(self.cross_entropy, feed_dict=feed_dict)))

            # 保存RNN模型
            logger.info('Saving RNN model...')
            self.saver.save(sess, self.save_model_path)

    # 对新数据进行预测
    def predict(self, data):
        with tf.Session() as sess:
            logger.info('Restoring RNN model...')
            self.saver.restore(sess, self.save_model_path)
            predict = sess.run(self.final_output, feed_dict={self._inputs: data})
        return predict