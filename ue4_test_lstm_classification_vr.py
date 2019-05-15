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
input_data_column_cnt = 1 * 6  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

seq_length = 240  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
forget_bias = 1.0  # 망각편향(기본값 1.0)
num_stacked_layers = 1  # stacked LSTM layers 개수
keep_prob = 1.0  # dropout할 때 keep할 비율

epoch_num = 1000  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01  # 학습률

X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, output_data_column_cnt])
print("Y: ", Y)

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, output_data_column_cnt])
print("targets: ", targets)

predictions = tf.placeholder(tf.float32, [None, output_data_column_cnt])
print("predictions: ", predictions)


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




# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
save_file = './model/train_model.ckpt'
saver = tf.train.Saver()
# # Save the model
# saver.save(sess, save_file)
# print('Trained Model Saved.')

# with tf.Session() as sess:
sess = tf.Session()
saver.restore(sess, save_file)

# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
index = 0
outputx_list = []
outputy_list = []

# print(index + 119, end=", ")

# print("recent_data:", data, end=" : ")
# 내일 종가를 예측해본다

b = np.zeros((240, 6))
# b[0] = [1, 1, 1, 1, 1, 1]
test_predict = sess.run(hypothesis, feed_dict={X: [b]})

# print(test_predict[0][0], end=", ")
print(test_predict[0][0])
# if (test_predict > 0.8):
#     print("moving")
# else:
#     print("stand")
# test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
# print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다
# for data in dataX:
#     # print(index + 119, end=", ")
#
#     # print("recent_data:", data, end=" : ")
#     # 내일 종가를 예측해본다
#     test_predict = sess.run(hypothesis, feed_dict={X: dataX[index:index + 1]})
#
#     outputx_list.append(index + seq_length - 1)
#     # print(test_predict[0][0], end=", ")
#     outputy_list.append(test_predict[0][0])
#
#     # if (test_predict > 0.8):
#     #     print("moving")
#     # else:
#     #     print("stand")
#     index = index + 1
# # test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
# # print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다
# print(outputy_list)
# # Make a line plot: year on the x-axis, pop on the y-axis
# # plt.plot(outputx_list, outputy_list)
# # plt.show()
# # ~929 stand
# # ~2233 : moving
# # 3600 : stand
# # 4468 moving
# # 5248 stand
# # 6106 moving
# # 7138 stand
# # ~ end moving
