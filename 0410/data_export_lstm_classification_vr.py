import datetime
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
# tf.set_random_seed(777)

# 하이퍼파라미터
input_data_column_cnt = 1 * 3  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

seq_length = 960 * 2
# 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
forget_bias = 1.0  # 망각편향(기본값 1.0)
num_stacked_layers = 1  # stacked LSTM layers 개수
keep_prob = 1.0  # dropout할 때 keep할 비율

epoch_num = 100  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01  # 학습률

my_dict = {"제자리": 0, "움직임": 1}


def dataFilePreproccessing(data_folder_name):
    _folder_path = os.path.join(data_folder_name)
    for file_index in os.listdir(_folder_path):
        file_path = os.path.join(_folder_path, file_index)
        L = []

        file = open(file_path, 'r')

        lines = file.readlines()

        file.close()

        for line in lines:  # 마지막에 \n 이걸 없애주네
            L.append(line.split('\n')[0])
        total_list = []
        for line in L:

            my_list = line
            data_list = []

            # for i in range(len(my_list)):
            # if i > 1:
            #     break

            my_list = my_list.strip()
            temp_list = my_list.split(" ")
            for temp in temp_list:
                data_list.append(temp.split("=")[1])
            total_list.append(data_list)
            # print("")
        x_np = np.zeros(0).reshape(0, input_data_column_cnt)
        x_np = np.append(x_np, np.asarray(total_list), axis=0)
        for _i in range(0, len(x_np) - seq_length):
            _x = x_np[_i: _i + seq_length]
            # _y = y_np[i + seq_length]  # 다음 나타날 주가(정답)
            # if i is 0:
            #     print(_x, "->", _y)  # 첫번째 행만 출력해 봄
            dataX.append(_x)  # dataX 리스트에 추가
            # dataY.append(_y)  # dataY 리스트에 추가
            dataY.append([my_dict[folder_path.split('/')[-1]]])


folder_path = "./data/vr_0410/제자리"

dataX = []  # 입력으로 사용될 Sequence Data
dataY = []  # 출력(타켓)으로 사용

dataFilePreproccessing(folder_path)
num = 1024
split_num = (len(dataX) // num + 1)
tempX = np.array_split(dataX, split_num)
tempY = np.array_split(dataY, split_num)

for i in range(split_num):
    np.savez("./npy_data/" + folder_path.split('/')[-1] + "/%d.npz" % (i + 51), x=tempX[i], y=tempY[i])
    # np.savez("D:/npy_data/" + folder_path.split('/')[-1] + "/%d.npz" % i, x=tempX[i], y=tempY[i])
print("end")

# np.save("D:/npy_data/ohX.npy", dataX)
# np.save("D:/npy_data/ohY.npy", dataY)
