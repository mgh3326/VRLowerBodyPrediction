import numpy as np
import matplotlib.pyplot as plt

input_data_column_cnt = 3 * 6  # 입력데이터의 컬럼 개수(Variable 개수)


def dataFilePreproccessing(file_path):
    L = []
    file = open(file_path, 'r')

    lines = file.readlines()

    file.close()

    for line in lines:  # 마지막에 \n 이걸 없애주네
        L.append(line.split('\n')[0])
    for line in L:
        my_list = line.split("//")
        data_list = []

        for i in range(len(my_list)):

            my_list[i] = my_list[i].strip()
            temp_list = my_list[i].split(" ")
            for temp in temp_list:
                data_list.append(temp.split("=")[1])
        total_list.append(data_list)


file_path = "./data/vr/제자리/제자리.txt"
total_list = []
dataFilePreproccessing(file_path)
x_np = np.zeros(0).reshape(0, input_data_column_cnt)
x_np = np.append(x_np, np.asarray(total_list), axis=0)
# oh_np = np.diff(x_np[:, 5])

np.savetxt("foo.csv", x_np, delimiter=",", fmt="%s")
# plt.plot(x_np[:, 0], label="Head_Rotate_P")
# plt.plot(x_np[:, 1], label="Head_Rotate_Y")
# plt.plot(x_np[:, 2], label="Head_Rotate_R")
# plt.show()
# plt.savefig('books_read.png')
