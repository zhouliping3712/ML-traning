# -*- coding: utf-8 -*-
"""
数字字母识别
利用RNN对验证码的数据集进行多分类
"""
from TensorFlow_RNN import RNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

CSV_FILE_PATH = 'F://验证码识别/data.csv'          # CSV 文件路径
df = pd.read_csv(CSV_FILE_PATH)                   # 读取CSV文件

# 数据集的特征
features = ['v'+str(i+1) for i in range(16*20)]
print(features.shape)

raise TypeError()
labels = df['label'].unique()
# 对样本的真实标签进行标签二值化
lb = LabelBinarizer()
lb.fit(labels)
y_ture = pd.DataFrame(lb.transform(df['label']), columns=['y'+str(i) for i in range(31)])
y_bin_columns = list(y_ture.columns)

for col in y_bin_columns:
    df[col] = y_ture[col]

# 将数据集分为训练集和测试集，训练集70%, 测试集30%
x_train, x_test, y_train, y_test = train_test_split(df[features], df[y_bin_columns], \
                                                    train_size = 0.7, test_size=0.3, random_state=123)

# 构建RNN网络
# 模型保存地址
MODEL_SAVE_PATH = 'logs/RNN_train.ckpt'
# RNN初始化
element_size = 16
time_steps = 20
num_classes = 31
hidden_layer_size = 300
batch_size = 960

new_x_train = np.array(x_train).reshape((-1, time_steps, element_size))
new_x_test = np.array(x_test).reshape((-1, time_steps, element_size))

rnn = RNN(element_size=element_size,
          time_steps=time_steps,
          num_classes=num_classes,
          batch_size=batch_size,
          hidden_layer_size= hidden_layer_size,
          epoch=1000,
          save_model_path=MODEL_SAVE_PATH,
          )

# 训练RNN
rnn.train(new_x_train, y_train)
# 预测数据
y_pred = rnn.predict(new_x_test)

# 预测分类
label = '123456789'
prediction = []
for pred in y_pred:
    label = labels[list(pred).index(max(pred))]
    prediction.append(label)

# 计算预测的准确率
x_test['prediction'] = prediction
x_test['label'] = df['label'][y_test.index]
print(x_test.head())
accuracy = accuracy_score(x_test['prediction'], x_test['label'])
print('CNN的预测准确率为%.2f%%.'%(accuracy*100))