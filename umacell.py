import pandas as pd
import math
import cmath
import csv
import numpy as np
import matplotlib.pyplot as plt

# 中心工作频率和波长
c = 3 * 10 ** 8  # 光速
f = 9 * 10 ** 8  # 工作中心频率 12.5GHz
lamb = c / f  # 中心频率波长
k = 2 * math.pi / lamb  # 相移常数 300=3*10^8/1*10^6 对频率作了处理

#  阵元参数
G = lamb / 2  # 间隔11.5mm
phi0 = 0  # 方位角方向
theta0 = 0  # 仰角方向。共同构成主波束方向。这边为deg，需要*np.pi/180转换为弧度
row = 8 # 阵元行数
column = 8  # 阵元列数
delta = k * G * np.sin(theta0*np.pi/180)  # 每个阵元在仰角条件下的理想相位

def read_csv(csvfile):
    data_x = []  # 初始化空x、y列表
    data_y = []  # 没归一
    csv_data = pd.read_csv(csvfile) # 同文件夹下的data文件，读取单个阵元数据
    n_data = len(csv_data)  # 定义数据长度（循环总长）为从文件中获得数据条数
    # 将表格文件转为两个有映射关系的一维向量：第0列为，第1列为
    for i in range(0, n_data):  # 0循环到n_data-1，共n_data次
        data_x.append(csv_data.loc[i][0])  # x列表中插入文件第i行第0列的数据：扫描角度
        data_y.append(csv_data.loc[i][1])
    return data_x,data_y, n_data

def parameter_initial():  # 基本初始数据计算和获取
    position = []  # 点源位置
    for i in range(0, row):  # 这边获得的已经是8*8
        for j in range(0, column):
            position.append([i*G, j*G])  # 最终获得一个
    return position

def radiation(data_x, data_y, n_data):
    position= parameter_initial()  #根据初始给定值获得坐标和激励矩阵
    phase =  np.zeros((row*column , 1))
    csvfile2 = open("./dataset/umaresult.csv", "w+", newline='')  # 文件操作：（新建）并打开result.csv，写入模式
    writer = csv.writer(csvfile2)
    data_array = []  # 初始化数据列表
    max_data = 0
    # 遗留的问题：是否应该比较归一化，还是比较绝对值→副瓣低某种程度也能说明更多能量在主瓣...？
    data_new = []  # 初始化new列表
    # # 从db数据中恢复出原值
    contributor = []
    for i in range(0, n_data):
        contributor.append(10**(data_y[i]/10))  # 分贝值和实际值之间的换算，y2是没归一化的
    start = 0
    for i in range(start, n_data-start):  # 生成数据长度和hfss得到的csv文件长度保持一致（才能做对比）
        a = complex(0, 0)  # 初始化a为复数0+j0。对所扫描的某个角度。每个都需要重新初始化a
        k_d = k * (np.sin(data_x[i] * np.pi / 180) - np.sin(theta0 * np.pi / 180))  # 1*n_data的向量
        for j in range(0,row*column):
            zeta = k_d * (position[j][0] * np.cos(phi0) + position[j][1] * np.sin(phi0))
            a = a + contributor[i] * np.exp(complex(0, (phase[j][0] * np.pi/180  + zeta)))
        data_new.append(abs(a))
    # print('data_new:',data_new)
    max_data = np.max(data_new)
    for i in range(start, n_data - start):
        data_array.append(20 * math.log10(data_new[i]/max_data))
    print('data_array:',data_array)

    # list=[]
    # list.append(data_x)
    # list.append(data_array)
    writer.writerows([data_x,data_array])

    # second_idx, second_value = get_sll(data_array)
    # print('副瓣值为：', second_value)

    plt.plot(data_x[start:n_data-start], data_array, "y")  # 画出现在计算得到的方向图
    plt.show()  # 显示图像


def get_sll(gain):
    psb_idx = []
    psb_num = []
    arr = np.array(gain)
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
    return second_idx, arr[psb_idx[second_idx]]

if __name__ == '__main__':
    csvfile1 = "./dataset/umacell.csv"
    data_x, data_y, n_data=read_csv(csvfile1)
    print(data_x)
    radiation(data_x, data_y, n_data)



