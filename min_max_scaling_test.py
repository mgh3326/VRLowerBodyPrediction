import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
tf.set_random_seed(777)


# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차원


# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


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
output_list = my_list
print("Hello")
