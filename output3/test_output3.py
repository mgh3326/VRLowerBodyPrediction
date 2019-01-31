import os

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
tf.set_random_seed(777)

# 하이퍼파라미터
input_data_column_cnt = 3 * 6  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 3 * 6  # 결과데이터의 컬럼 개수

seq_length = 24  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
forget_bias = 1.0  # 망각편향(기본값 1.0)
num_stacked_layers = 1  # stacked LSTM layers 개수
keep_prob = 1.0  # dropout할 때 keep할 비율

epoch_num = 100  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01  # 학습률


class Rotation:
    x = ""
    y = ""
    z = ""
    w = ""


class Position:
    x = ""
    y = ""
    z = ""


class Model:
    name = ""
    rotation = Rotation()
    position = Position()

    def input_roation_position(self, a_list):
        for i in range(len(a_list)):
            if a_list[i] == "":
                a_list[i] = 0
        self.rotation.x = float(a_list[0])
        self.rotation.y = float(a_list[1])
        self.rotation.z = float(a_list[2])
        self.rotation.w = float(a_list[3])
        self.position.x = float(a_list[4])
        self.position.y = float(a_list[5])
        self.position.z = float(a_list[6])


class ListModel:
    time = ""
    models = []  # 모델이 들어갈 리스트


import csv
from collections import OrderedDict

X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
filename = "../preprocesiing_run_Take_2019-01-18_05.11.33_PM.csv"
import pandas as pd

name = [
    "Time", "Head.X_worldpos", "Head.Y_worldpos", "Head.Z_worldpos", "LeftHand.X_worldpos", "LeftHand.Y_worldpos",
    "LeftHand.Z_worldpos", "RightHand.X_worldpos", "RightHand.Y_worldpos", "RightHand.Z_worldpos", "Head.X_rotations",
    "Head.Y_rotations", "Head.Z_rotations", "LeftHand.X_rotations", "LeftHand.Y_rotations", "LeftHand.Z_rotations",
    "RightHand.X_rotations", "RightHand.Y_rotations", "RightHand.Z_rotations", "Hips.X_worldpos", "Hips.Y_worldpos",
    "Hips.Z_worldpos", "LeftFoot.X_worldpos", "LeftFoot.Y_worldpos", "LeftFoot.Z_worldpos", "RightFoot.X_worldpos",
    "RightFoot.Y_worldpos", "RightFoot.Z_worldpos", "Hips.X_rotations", "Hips.Y_rotations", "Hips.Z_rotations",
    "LeftFoot.X_rotations", "LeftFoot.Y_rotations", "LeftFoot.Z_rotations", "RightFoot.X_rotations",
    "RightFoot.Y_rotations", "RightFoot.Z_rotations"]
raw_dataframe = pd.read_csv(filename, names=name)
input_name = name[1:19]
input_df = raw_dataframe[input_name]
x_np = input_df.values[1:].astype(np.float)  # 금액&거래량 문자열을 부동소수점형으로 변환한다


# 모델(LSTM 네트워크) 생성
def lstm_cell():
    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  to the biases of the forget gate
    #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)

    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)

# [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
# 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

save_file = './model/train_model.ckpt'
saver = tf.train.Saver()
# Save the model


# with tf.Session() as sess:
sess = tf.Session()
saver.restore(sess, save_file)

# sequence length만큼의 가장 최근 데이터를 슬라이싱한다


recent_data = np.array([x_np[len(x_np) - seq_length:]])
print("recent_data.shape:", recent_data.shape)
print("recent_data:", recent_data)

# 내일 종가를 예측해본다
test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

print("test_predict", test_predict[0])
# test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
# print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다
