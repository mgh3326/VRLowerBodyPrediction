import json
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
input_data_column_cnt = 3 * 3  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

seq_length = 100  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
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
status_list = ["stand", "walk"]
my_list = []
with open('./data/test_0515/test.txt', 'r') as f:
    for line in f.readlines():
        jsonInput = json.loads(line)
        temp_list = [jsonInput["head"]["pitch"], jsonInput["head"]["yaw"], jsonInput["head"]["roll"],
                     jsonInput["lHand"]["pitch"], jsonInput["lHand"]["yaw"], jsonInput["lHand"]["roll"],
                     jsonInput["rHand"]["pitch"], jsonInput["rHand"]["yaw"], jsonInput["rHand"]["roll"]]
        my_list.append(temp_list)
        # print(line)
x_np = np.zeros(0).reshape(0, input_data_column_cnt)
x_np = np.append(x_np, np.asarray(my_list), axis=0)
dataX = []
for i in range(0, len(x_np) - seq_length):
    _x = x_np[i: i + seq_length]
    # _y = y_np[i + seq_length]  # 다음 나타날 주가(정답)
    # if i is 0:
    #     print(_x, "->", _y)  # 첫번째 행만 출력해 봄
    dataX.append(_x)  # dataX 리스트에 추가


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

# 손실함수로 평균제곱오차를 사용한다
loss = tf.reduce_sum(tf.square(hypothesis - Y))
# 최적화함수로 AdamOptimizer를 사용한다
optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.RMSPropOptimizer(learning_rate) # LSTM과 궁합 별로임

train = optimizer.minimize(loss)

# RMSE(Root Mean Square Error)
# 제곱오차의 평균을 구하고 다시 제곱근을 구하면 평균 오차가 나온다
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions))) # 아래 코드와 같다
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

train_error_summary = []  # 학습용 데이터의 오류를 중간 중간 기록한다
test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다

sess = tf.Session()
sess.run(tf.global_variables_initializer())

save_file = './model_0508/train_model.ckpt'
saver = tf.train.Saver()
# Save the model

# with tf.Session() as sess:
saver.restore(sess, save_file)

# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
index = 0
outputx_list = []
outputy_list = []

# print(index + 119, end=", ")

# print("recent_data:", data, end=" : ")
# 내일 종가를 예측해본다
# oh = dataX[0]
# test_predict = sess.run(hypothesis, feed_dict={X: [dataX[0]]})
#
# # print(test_predict[0][0], end=", ")
# print(test_predict[0][0])
# b = np.zeros((240, 6))
# if (test_predict > 0.8):
#     print("moving")
# else:
#     print("stand")
# test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
# print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다
for data in dataX:
    # print(index + 119, end=", ")

    # print("recent_data:", data, end=" : ")
    # 내일 종가를 예측해본다
    test_predict = sess.run(hypothesis, feed_dict={X: dataX[index:index + 1]})

    outputx_list.append(index + seq_length - 1)
    # print(test_predict[0][0], end=", ")
    result = test_predict[0][0]
    # if result > 0.5:
    #     result = 1
    # else:
    #     result = 0
    outputy_list.append(result)
    # if result > 0.5:
    #     outputy_list.append(1)
    #
    # else:
    #     outputy_list.append(0)

    # if (test_predict > 0.8):
    #     print("moving")
    # else:
    #     print("stand")
    index = index + 1
# test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
# print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다
print(outputy_list)
# ~ 209 : s
# ~ 328 : w
# ~ 507 : s
# ~ 612 : w
# ~ 690 : s
jump_num = 46
# temp_list = [209, 328, 507, 612, 690]
temp_list = [209+jump_num , 328+jump_num , 507+jump_num , 612+jump_num , 690+jump_num ]
eval_list = []
status = 0

for i in range(temp_list[0] - (seq_length - 1)):
    eval_list.append(0)
for i in range(temp_list[1] - temp_list[0]):
    eval_list.append(1)
for i in range(temp_list[2] - temp_list[1]):
    eval_list.append(0)
for i in range(temp_list[3] - temp_list[2]):
    eval_list.append(1)
# for i in range(temp_list[4] - temp_list[3]):
#     eval_list.append(0)
for i in range(len(outputy_list) - len(eval_list)):
    eval_list.append(0)
count = 0
for i in range(len(eval_list)):
    if eval_list[i] != outputy_list[i]:
        count += 1
print(len(eval_list), count, (len(eval_list) - count) / len(eval_list))
# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(outputx_list, outputy_list)
# plt.plot(outputx_list, eval_list)
plt.show()
# # ~175 stand
# # ~252 : moving
# #  : stand


# switch_index_list = [100, 200, 300]
# current_status = 0
# temp_status_list = [0]
#
# for switch_index in switch_index_list:
#     for i in range(switch_index):
#         temp_status_list.append(current_status)
#     if current_status == 0:
#         current_status = 1
#     else:
#         current_status = 0
