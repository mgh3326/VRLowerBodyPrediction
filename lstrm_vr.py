import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
tf.set_random_seed(777)

# 하이퍼파라미터
input_data_column_cnt = 3 * 7  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 18 * 7  # 결과데이터의 컬럼 개수

seq_length = 240  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
forget_bias = 1.0  # 망각편향(기본값 1.0)
num_stacked_layers = 1  # stacked LSTM layers 개수
keep_prob = 1.0  # dropout할 때 keep할 비율

epoch_num = 1000  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
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

skeleton_list = []  # 스켈레톤 순서를 담기 위함
oh_list = ["Rotation", "Position"]
oh2_list = ["X", "Y", "Z", "W", "X", "Y", "Z"]
my_list = []
with open('data/Take 2018-12-19 02.30.17 PM.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print(spamreader.line_num)
        # print(', '.join(row))
        if spamreader.line_num == 4:  # 스켈레톤 순서를 담기 위함
            temp_list = []
            for i in row:
                if i != "":
                    temp_list.append(str(i).split("_")[-1])  # 앞에 스켈레톤 번호는 지우기 위함
            skeleton_list = list(OrderedDict.fromkeys(temp_list))
            print("스켈레톤 순서를 담기 위함")
        elif spamreader.line_num >= 8:  # position rotation 값을 넣어주자
            list_model = ListModel()
            list_model.models = []
            list_model.time = row[1]  # 시간
            index = 0  # 0 :x  1 : y 2 :z 3 : w 4 :x 5 :y 6: z
            for i in range(2, len(row), 7):
                temp_model = Model()
                temp_model.name = skeleton_list[index]
                temp_model.position = Position()  # 이 과정이 왜 필요한거지
                temp_model.rotation = Rotation()

                index += 1
                a_list = row[i:i + 7]
                temp_model.input_roation_position(a_list)
                list_model.models.append(temp_model)
            my_list.append(list_model)
            # 마지막 꺼가 빠져서 임의로 넣도록 하자
input_list = []
output_list = []
for i in my_list:
    temp_list = ListModel()
    temp_list.time = i.time
    temp_model = Model()
    temp_list.models = []
    temp_list.models.append(i.models[4])
    temp_list.models.append(i.models[8])
    temp_list.models.append(i.models[12])
    i.models.remove(i.models[12])
    i.models.remove(i.models[8])
    i.models.remove(i.models[4])

    input_list.append(temp_list)
output_list = my_list.copy()
my_list.clear()
print("Hello")
x_np = np.zeros(0)
y_np = np.zeros(0)
for j in range(len(input_list)):
    # input용
    if j is 0:
        for k in range(seq_length - 1):  # 0 번째 값을 238번까지 넣는 과정
            x_list = []
            for i in range(len(input_list[j].models)):
                temp_list = [input_list[j].models[i].rotation.x, input_list[j].models[i].rotation.y,
                             input_list[j].models[i].rotation.z, input_list[j].models[i].rotation.w,
                             input_list[j].models[i].position.x, input_list[j].models[i].position.y,
                             input_list[j].models[i].position.z]
                x_list.extend(temp_list)
            temp_np = np.asarray(x_list)
            x_np = np.append(x_np, temp_np)
            # output용
            y_list = []
            for i in range(len(output_list[j].models)):
                temp_list = [output_list[j].models[i].rotation.x, output_list[j].models[i].rotation.y,
                             output_list[j].models[i].rotation.z, output_list[j].models[i].rotation.w,
                             output_list[j].models[i].position.x, output_list[j].models[i].position.y,
                             output_list[j].models[i].position.z]
                y_list.extend(temp_list)
            temp_np = np.asarray(y_list)
            y_np = np.append(y_np, temp_np)
    x_list = []

    for i in range(len(input_list[j].models)):
        temp_list = [input_list[j].models[i].rotation.x, input_list[j].models[i].rotation.y,
                     input_list[j].models[i].rotation.z, input_list[j].models[i].rotation.w,
                     input_list[j].models[i].position.x, input_list[j].models[i].position.y,
                     input_list[j].models[i].position.z]
        x_list.extend(temp_list)
    temp_np = np.asarray(x_list)
    x_np = np.append(x_np, temp_np)
    # output용
    y_list = []
    for i in range(len(output_list[j].models)):
        temp_list = [output_list[j].models[i].rotation.x, output_list[j].models[i].rotation.y,
                     output_list[j].models[i].rotation.z, output_list[j].models[i].rotation.w,
                     output_list[j].models[i].position.x, output_list[j].models[i].position.y,
                     output_list[j].models[i].position.z]
        y_list.extend(temp_list)
    temp_np = np.asarray(y_list)
    y_np = np.append(y_np, temp_np)
x_np = x_np.reshape((-1, 21))  # reshpae 성공
y_np = y_np.reshape((-1, 7 * 18))  # reshpae 성공
dataX = []  # 입력으로 사용될 Sequence Data
dataY = []  # 출력(타켓)으로 사용

for i in range(0, len(y_np) - seq_length):
    _x = x_np[i: i + seq_length]
    _y = y_np[i + seq_length]  # 다음 나타날 주가(정답)
    if i is 0:
        print(_x, "->", _y)  # 첫번째 행만 출력해 봄
    dataX.append(_x)  # dataX 리스트에 추가
    dataY.append(_y)  # dataY 리스트에 추가
print("Hello")
# 학습용/테스트용 데이터 생성
# 전체 70%를 학습용 데이터로 사용
train_size = int(len(dataY) * 0.7)
# 나머지(30%)를 테스트용 데이터로 사용
test_size = len(dataY) - train_size

# 데이터를 잘라 학습용 데이터 생성
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

# 데이터를 잘라 테스트용 데이터 생성
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 7 * 18])
print("Y: ", Y)

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 7 * 18])
print("targets: ", targets)

predictions = tf.placeholder(tf.float32, [None, 7 * 18])
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
test_predict = ''  # 테스트용데이터로 예측한 결과

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습한다
start_time = datetime.datetime.now()  # 시작시간을 기록한다
print('학습을 시작합니다...')
for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
        # 학습용데이터로 rmse오차를 구한다
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)

        # 테스트용데이터로 rmse오차를 구한다
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)

        # 현재 오류를 출력한다
        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1, train_error, test_error,
                                                                                 test_error - train_error))

end_time = datetime.datetime.now()  # 종료시간을 기록한다
elapsed_time = end_time - start_time  # 경과시간을 구한다
print('elapsed_time:', elapsed_time)
print('elapsed_time per epoch:', elapsed_time / epoch_num)

# 하이퍼파라미터 출력
print('input_data_column_cnt:', input_data_column_cnt, end='')
print(',output_data_column_cnt:', output_data_column_cnt, end='')

print(',seq_length:', seq_length, end='')
print(',rnn_cell_hidden_dim:', rnn_cell_hidden_dim, end='')
print(',forget_bias:', forget_bias, end='')
print(',num_stacked_layers:', num_stacked_layers, end='')
print(',keep_prob:', keep_prob, end='')

print(',epoch_num:', epoch_num, end='')
print(',learning_rate:', learning_rate, end='')

print(',train_error:', train_error_summary[-1], end='')
print(',test_error:', test_error_summary[-1], end='')
print(',min_test_error:', np.min(test_error_summary))

# 결과 그래프 출력
plt.figure(1)
plt.plot(train_error_summary, 'gold')
plt.plot(test_error_summary, 'b')
plt.xlabel('Epoch(x100)')
plt.ylabel('Root Mean Square Error')

plt.figure(2)
plt.plot(testY, 'r')
plt.plot(test_predict, 'b')
plt.xlabel('Time Period')
plt.ylabel('Stock Price')
plt.show()

# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
recent_data = np.array([x_np[len(x_np) - seq_length:]])
print("recent_data.shape:", recent_data.shape)
print("recent_data:", recent_data)

# 내일 종가를 예측해본다
test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

print("test_predict", test_predict[0])
# test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
# print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다
