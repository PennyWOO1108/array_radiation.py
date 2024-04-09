import numpy as np
import pandas as pd
import math
import cmath
import csv

def read_csv():
    gain = []  # 初始化空x、y列表
    csv_data = pd.read_csv("./dataset/array88.csv") # 同文件夹下的data文件，读取单个阵元数据
    # n_data = len(csv_data)  # 定义数据长度（循环总长）为从文件中获得数据条数
    n_data= 360
    for i in range(120,240):  # 0循环到n_data-1，共n_data次
        gain.append(csv_data.loc[i][3])  # 没归一化
    # print("总长度为", n_data, "各角度增益为", gain)
    return gain

def get_random_matrix():
    matrix = 90 * np.random.randint(1,5,(8,8))
    return matrix

def get_sll(gain):
    psb_idx = []
    psb_num = []
    arr = np.array(gain)
    # print(arr,arr.size)
    for i in range(arr.size):
        if i == 0 and arr[i] > arr[arr.size - 1] and arr[i] < arr[i + 1]:
            psb_idx.append(i)
            psb_num.append(arr[i])
        elif i == arr.size - 1 and arr[i] > arr[0] and arr[i] > arr[i - 1]:
            psb_idx.append(i)
            psb_num.append(arr[i])
        elif i != 0 and i != arr.size-1 and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            psb_idx.append(i)
            psb_num.append(arr[i])
        else:
            continue
    print(psb_idx, psb_num)
    max_idx = psb_num.index(max(psb_num))
    print('最大值为', max(psb_num))
    psb_num.pop(max_idx)
    psb_idx.pop(max_idx)
    second_idx = psb_num.index(max(psb_num))
    # print('副瓣值为：', arr[psb_idx[second_idx]])
    return second_idx, arr[psb_idx[second_idx]]

if __name__ == '__main__':
    gain = read_csv()
    # gain  = np.random.randint(500, size=50)
    second_idx, second_value = get_sll(gain)
    print('副瓣值为：', second_value)