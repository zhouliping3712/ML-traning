import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import os
import random
import numpy as np
from PIL import Image


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


class RNN2captcha(object):
    # x > 16 160 40
    # y > 40
    def __init__(self):
        self.train_list = os.listdir("img/")
        self.test_list = []

        test_count = int(len(self.train_list)*0.1)

        for i in range(test_count):
            img = random.choice(self.train_list)
            self.train_list.remove(img)
            self.test_list.append(img)

        print("Train count: {}".format(len(self.train_list)))
        print("Test count: {}".format(len(self.test_list)))

        self.image_height = 60
        self.image_width = 100
        self.max_captcha = 4
        self.char_set_len = 10
        self.train_img_path = "img"
        self.char_set = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        self.model_save_dir = "model/rnnModel"

        self.dimhidden = 128

    @staticmethod
    def gen_captcha_text_image(img_path, img_name):
        """
        返回一个验证码的array形式和对应的字符串标签
        :return:tuple (str, numpy.array)
        """
        # 标签
        label = img_name.split("_")[0]
        # 文件
        img_file = os.path.join(img_path, img_name)
        captcha_image = Image.open(img_file)

        captcha_image = captcha_image.convert("L")

        captcha_array = np.array(captcha_image)  # 向量化
        return label, captcha_array

    @staticmethod
    def convert2gray(img):
        """
        图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
        :param img:
        :return:
        """
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def text2vec(self, text):
        """
        转标签为oneHot编码
        :param text: str
        :return: numpy.array
        """
        text_len = len(text)
        if text_len > self.max_captcha:
            raise ValueError('验证码最长{}个字符'.format(self.max_captcha))

        vector = np.zeros(self.max_captcha * self.char_set_len)

        for i, ch in enumerate(text):
            idx = i * self.char_set_len + self.char_set.index(ch)
            vector[idx] = 1
        return vector

    def get_batch(self, n, size=16, mode="train"):
        batch_x = np.zeros([size, self.image_height * self.image_width])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化

        if mode == "train":

            max_batch = int(len(self.train_list) / size)
            # print(max_batch)
            if max_batch - 1 < 0:
                raise TypeError("训练集图片数量需要大于每批次训练的图片数量")
            if n > max_batch - 1:
                n = n % max_batch
            s = n * size
            e = (n + 1) * size
            this_batch = self.train_list[s:e]

            # this_batch = random.choices(self.train_list, k=size)
        elif mode == "test":
            max_batch = int(len(self.test_list) / size)
            # print(max_batch)
            if max_batch - 1 < 0:
                raise TypeError("训练集图片数量需要大于每批次训练的图片数量")
            if n > max_batch - 1:
                n = n % max_batch
            s = n * size
            e = (n + 1) * size
            this_batch = self.test_list[s:e]

            # this_batch = random.choices(self.test_list, k=size)
        else:
            raise TypeError("just input train or test")

        for i, img_name in enumerate(this_batch):
            label, image_array = self.gen_captcha_text_image(self.train_img_path, img_name)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            # print(img_name)
            # print(image_array)
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        # print(batch_x.shape, batch_y.shape)
        # print("1>>>>>>", batch_x[0])
        # print("2>>>>>>", batch_x[1])
        # print("===")
        batch_x = np.array(batch_x).reshape((size, self.image_height, self.image_width))
        # print(batch_x)
        return batch_x, batch_y

    def start_train(self):

        # 设置参数，权重，偏置

        W = {
            "h1": tf.Variable(tf.random_normal([self.image_width, self.dimhidden])),  # [28, 128]
            "h2": tf.Variable(tf.random_normal([self.dimhidden, self.max_captcha * self.char_set_len]))  # [128, 10]
        }

        b = {
            "b1": tf.Variable(tf.random_normal([self.dimhidden])),  # [128]
            "b2": tf.Variable(tf.random_normal([self.max_captcha * self.char_set_len]))  # [10]
        }

        learning_rate = 0.001
        x = tf.placeholder("float", [None, self.image_height, self.image_width])  # [count, 40, 160]
        y = tf.placeholder("float", [None, self.max_captcha * self.char_set_len])  # [count, 40]

        X = tf.reshape(x, [-1, self.image_width])  # > [N*60, 100]
        H_1 = tf.matmul(X, W["h1"]) + b["b1"]  # (N*60, 128) = (N*60, 100) * (100, 128) + (128)
        H_1 = tf.split(H_1, self.image_height, 0)  # [(N, 128), (N, 128), ...] length is 60
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.dimhidden, forget_bias=1.0)  # 128
        LSTM_O, LSTM_S = rnn.static_rnn(lstm_cell, H_1, dtype=tf.float32)  # [(N, 128)]
        pred = tf.matmul(LSTM_O[-1], W["h2"]) + b["b2"]  # (N, 10) = (N, 128) * (128, 10) + (10)

        # 交叉熵损失
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        # 梯度下降
        optm = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
        # 准确率
        predict = tf.reshape(pred, [-1, self.max_captcha, self.char_set_len])  # 预测结果  [16, 4, 10]
        max_idx_p = tf.argmax(predict, 2)  # 预测结果
        max_idx_l = tf.argmax(tf.reshape(y, [-1, self.max_captcha, self.char_set_len]), 2)  # 标签
        # 计算准确率
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            batch_size = 10
            display_step = 100
            total_train_times = 20000

            init = tf.global_variables_initializer()
            print("Network Ready!")
            sess.run(init)

            # 恢复模型
            if os.path.exists(self.model_save_dir):
                try:
                    print("加载模型")
                    saver.restore(sess, self.model_save_dir)
                # 判断捕获model文件夹中没有模型文件的错误
                except ValueError:
                    print("model文件夹为空，将创建新模型")
            else:
                pass

            print("Start optimization")

            for i in range(total_train_times):
                batch_xs, batch_ys = self.get_batch(i, size=batch_size, mode="train")
                feeds = {x: batch_xs, y: batch_ys}
                sess.run(optm, feed_dict=feeds)

                if i % display_step == 0:
                    batch_xs, batch_ys = self.get_batch(i, size=batch_size, mode="train")
                    print("Input x: {} y: {}".format(batch_xs.shape, batch_ys.shape))

                    feeds = {x: batch_xs, y: batch_ys}
                    cost_o, accr_o, pred_p, pred_l = sess.run([cost, accuracy_char_count, pred, max_idx_l], feed_dict=feeds)

                    # print(pred_p.shape)
                    # print(pred_p)
                    # print(pred_l.shape)
                    # print(pred_l)

                    print(list(pred_p[0][:20]))
                    print(list(pred_p[1][:20]))
                    # print("================")

                    print("Train: {}/{} cost: {} accr: {}".format(i, total_train_times, cost_o, accr_o))

                    # batch_xs, batch_ys = self.get_batch(1, size=batch_size, mode="test")
                    # feeds = {x: batch_xs, y: batch_ys}
                    # cost_o, accr_o = sess.run([cost, accuracy_char_count], feed_dict=feeds)
                    # print("Test: {}/{} cost: {} accr: {}".format(i, total_train_times, cost_o, accr_o))

                if i % 500 == 0:
                    saver.save(sess, self.model_save_dir)
                    print("定时保存模型成功")

            print("Optimization Finished")


def main():
    rc = RNN2captcha()
    rc.start_train()


if __name__ == '__main__':
    main()