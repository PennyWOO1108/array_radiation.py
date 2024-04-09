import pandas as pd
import math
import cmath
import csv
import numpy as np
import matplotlib.pyplot as plt

# 中心工作频率和波长
c = 3 * 10 ** 8  # 光速
f = 12.5 * 10 ** 8  # 工作中心频率 12.5GHz
lamb = c / f  # 中心频率波长
k = 2 * math.pi / lamb  # 相移常数 300=3*10^8/1*10^6 对频率作了处理

#  阵元参数
G = 11.5 * 10 ** -3  # 间隔11.5mm
phi0 = 0  # 方位角方向
theta0 = 0  # 仰角方向。共同构成主波束方向。这边为deg，需要*np.pi/180转换为弧度
row = 8 # 阵元行数
column = 8 # 阵元列数
delta = k * G * np.sin(theta0*np.pi/180)  # 每个阵元在仰角条件下的理想相位

def calcuate_phase():
    phase =  [[0], [90], [90], [0], [90], [0], [90], [0], [0], [0], [90], [90], [90], [0], [0], [0]]
    ideal_phase = []
    new_phase = []
    for i in range(0,row):
        for j in range(0,column):
            idx = i * row + j  #假设1时，0*0+0，数组仅有1个数；当为8时，64个数
            ideal_phase.append(j * delta * 180/np.pi)  # 弧度转化成角度（假设希望主瓣方向在xoy平面上时，直接这样写，此时认为不同列波前不同相位）
            err = np.mod(ideal_phase[idx] - phase[idx][0],360)  #（计算理想相位【主波束方向增益最大】和当前相位的误差）
            if err >= 270 or err<=90:
                new_phase.append(phase[idx][0])  #当误差>=270或者<=90时，在1bit条件下我们认为当前的预置是好的
            elif err < 270 and err>90:
                new_phase.append(phase[idx][0] + 180)  #当误差<270或者>90时，在1bit条件下我们认为需要反向偏压
    print("预置相位:",phase)
    print("理论相位:",ideal_phase)
    print("最终相位:",new_phase)  #测试误差判断是否生效

    return new_phase #得到最终相位方案，用于绘制优化后方向图，为波束赋形效果提供参考

def read_csv(csvfile):  # 获取阵元数据
    data_x = []  # 初始化空x、y列表，初始时不采用归一化（负数处理需要转弯）
    # data_y1 = []  # 归一化
    data_y = []  # 没归一化
    csv_data = pd.read_csv(csvfile) # 同文件夹下的data文件，读取单个阵元数据
    #print(csv_data)
    n_data = len(csv_data)  # 定义数据长度（循环总长）为从文件中获得数据条数（函数认为表头不占行）
    # 将表格文件转为两个有映射关系的一维向量：第0列为，第1列为
    for i in range(0, n_data):  # 0循环到n_data-1，共n_data次
        data_x.append(csv_data.loc[i][0])  # x列表中插入文件第i行第0列的数据：扫描角度（精度为1°）
        # data_y1.append(csv_data.loc[i][1])  # y列表中插入文件第i行第1列的数据：归一化
        data_y.append(csv_data.loc[i][1])  # 非归一
    #print(data_x, len(data_x))
    #print(data_y, len(data_y))

    # plt.plot(data_x, data_y1, color='red')  # 画出归一化和非归一化读取得到的方向图
    # plt.plot(data_x, data_y, color='blue', linewidth=1)
    # plt.show()  # 显示图像
    return data_x,data_y, n_data

def parameter_initial():  # 阵列参数初始化

    position = []  # 天线元中心坐标
    amplititude = [] # 激励幅值
    phase = [] # 激励相位（是一种参考相位）

    for i in range(0, row):  # 这边获得的已经是8*8
        for j in range(0, column):
            position.append([i*G, j*G])  # 最终获得一个
            amplititude.append(1)  #采用T功分的馈网，理论上是均匀激励
            # phase.append(np.random.randint(0,4)*np.pi/2)  # 在预置和1bit作用下，最终有四种相位可能
            phase.append(0)  # 先测试最理想情况下
    # print("位置", position[7], "激励", amplititude[7])  #测试是否赋值成功
    return position

def radiation(data_x, data_y, n_data):  # 方向图函数
    position= parameter_initial()  #根据初始给定值获得坐标和激励矩阵
    phase =  np.zeros((row*column , 1))
    # phase = [[0], [0], [90], [0], [0], [0], [0], [90], [0], [90], [0], [0], [90], [0], [90], [0]]
    # phase = np.random.randint(0,2,(row*column,1))*90
    # phase = [[0], [270], [270], [180], [90], [0], [270], [180], [0], [0], [270], [90], [90], [0], [180], [180]]
    # print("位置矩阵是：", position, "幅值矩阵是：",phase )
    data_array = []  # 初始化储存方向图扫描点的列表
    max_data = 0
    # data_array.append(['Theta [deg]', '10Normalize [dB]'])  # 插入表头：扫描角度，方向图值（功率）
    # 遗留的问题：是否应该比较归一化，还是比较绝对值→副瓣低某种程度也能说明更多能量在主瓣...？
    data_new = []  # 初始化new列表
    # # 从db数据中恢复出原值
    contributor = []
    start = 0
    for i in range(start, n_data-start):
        contributor.append(10**(data_y[i]/10))  # 分贝值和实际值之间的换算，y2是没归一化的
    # print(contributor)  #打印出单个阵元在每个方向辐射强度的实际值
    for i in range(0, n_data):  # 生成数据长度和hfss得到的csv文件长度保持一致（才能做对比） n_data 表示 逐个求 n_data个角度上的增益
    #     data_array.append(data_x[i])  # 插入相应扫描角度
        a = complex(0, 0)  # 初始化a为复数0+j0。对所扫描的某个角度。每个都需要重新初始化a
        k_d = k * (np.sin(data_x[i] * np.pi / 180) - np.sin(theta0 * np.pi / 180))  # 1*n_data的向量，转角度为弧度
        # 参考prephase论文里的公式 阵因子 = exp(j*(预置相位，偏置相位，分布相位））
        for j in range(0,row*column):  # 综合每个点源在此的增益
            zeta = k_d * (position[j][0] * np.cos(phi0) + position[j][1] * np.sin(phi0))
            a = a + contributor[i] * np.exp(complex(0, (phase[j][0] * np.pi/180  + zeta)))
        # data_array[i + 1].append(10 * math.log10(abs(a)))  # 不作归一化处理，得到标准方向图数值
        # data_array.append(10 * np.log10(abs(a)))  # 不作归一化处理，得到标准方向图数值
        data_new.append(abs(a))
    print('data_new:',data_new)
    max_data = np.max(data_new)
    for i in range(start, n_data - start):
        gain_normal=10 * math.log10(data_new[i]/max_data)
        data_array.append(gain_normal)  # 做归一化（不如一开始就归一化）
    print('data_array:',data_array)

    second_idx, second_value = get_sll(data_array)  # 查找副瓣
    print('副瓣值为：', second_value)

    plt.plot(data_x[start:n_data-start], data_array, "y")  # 画出现在计算得到的方向图
    plt.show()  # 显示图像

    # return second_value

def get_sll(gain):  #判断副瓣大小
    psb_idx = []  # 储存索引
    psb_num = []  # 储存数值
    arr = np.array(gain)  #转为array方便使用后续操作
    # print(arr,arr.size)
    for i in range(arr.size):  #逐个寻找是局部最大值的数，再从符合条件的这些数中找第二大的数
        if i == 0 and arr[i] > arr[arr.size - 2] and arr[i] > arr[i + 1]:  # 列表中第一个数，如果大于最后一个数且大于第二个数
            psb_idx.append(i) #记忆索引
            psb_num.append(arr[i]) #记忆增益
        elif i == arr.size - 1 and arr[i] > arr[1] and arr[i] > arr[i - 1]:  # 列表中最后一个数，如果大于第一个数且大于倒数第二个数
            psb_idx.append(i)
            psb_num.append(arr[i])
        elif i != 0 and i != arr.size-1 and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:  #其他数值，判断同时大于前后即可
            psb_idx.append(i)
            psb_num.append(arr[i])
        # else:
        #     continue
    print(psb_idx, psb_num) #满足条件的所有数
    max_idx = psb_num.index(max(psb_num))
    print('最大值为', max(psb_num))
    psb_num.pop(max_idx)
    psb_idx.pop(max_idx)  #删去对最大值的记忆
    second_idx = psb_num.index(max(psb_num))  # 剩下的最大即为副瓣
    # print('副瓣值为：', arr[psb_idx[second_idx]])
    return second_idx, arr[psb_idx[second_idx]]

if __name__ == '__main__':
    # calcuate_phase()
    csvfile = "dataset/newdata.csv"
    data_x, data_y, n_data=read_csv(csvfile)  # 获取阵元数据
    # # max_angle = int((n_data - 1) / 2 + theta0 / 2)
    # # print(max_angle, data_x[max_angle])
    radiation(data_x, data_y, n_data)
    # position, amplititude, phase = parameter_initial()
    # print(position, amplititude, phase)
    # print("未归一化的天线增益",data_y2)



