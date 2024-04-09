#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
代码实现预期效果：对少数量，中等数量，大数量阵列进行了综合；红色线由方向图乘积定理得到的，蓝色的由hfss综合得到，基本重合
增益和副瓣波形包络完全一致，只是副瓣的波峰略有差异。
'''
# 导入相关库
import pandas as pd
import math
import cmath
import csv
import matplotlib.pyplot as plt

# 定义方向图类
class Pattern:

    # 定义辐射函数
    def radiation(self):
        data_x = []  # 初始化空x、y列表
        data_y = []
        csv_data = pd.read_csv("./data.csv") # 同文件夹下的data文件，读取单个阵元数据
        n_data = len(csv_data)  # 定义数据长度（循环总长）为从文件中获得数据条数

        # 将表格文件转为两个有映射关系的一维向量：第0列为，第1列为
        for i in range(0, n_data):  # 0循环到n_data-1，共n_data次
            data_x.append(csv_data.loc[i][0])  # x列表中插入文件第i行第0列的数据
            data_y.append(csv_data.loc[i][1])  # y列表中插入文件第i行第1列的数据

        # 在python中初始化阵列参数
        n_cell = 9  # 阵元数目（此处为1*9线阵）
        f = 1.575  # 工作中心频率 MHz?
        position = [0, 94, 206, 281, 393, 475, 587, 683, 785]  # 等效点源位置
        power = [0.2, 0.8, 0.4, 0.3, 0.5, 0.9, 0.2, 0.7, 0.4]  # 激励幅值
        phase = [0, 82, 165, 201, 247, 229, 262, 305, 334]  # 激励相位
        k = 2 * math.pi * f / 300  # 相移常数 300=3*10^8/1*10^6 对频率作了处理

        csvfile = open("newdata.csv", "w+", newline='')  #文件操作：（新建）并打开newdata.csv，写入模式
        writer = csv.writer(csvfile)
        data_array = []  # 初始化数据列表
        data_array.append(['Theta [deg]','10Normalize [dB]'])  # 插入表头：扫描角度，方向图值（功率）
        data_new = []  # 初始化new列表

        for i in range(0, n_data):  # 生成数据长度和hfss得到的csv文件长度保持一致（才能做对比）
            data_array.append([])
            data_array[i+1].append(data_x[i])  # 插入相应扫描角度
            a = complex(0, 0)  # 初始化a为复数0+j0
            k_d = k * math.sin(data_x[i] * math.pi / 180)  # 计算指数式除了位置以外的参数
            for j in range(0, n_cell):  # 遍历所有阵元，的到期共同作用的方向图结果
                a = a + power[j] * data_y[i] * cmath.exp(complex(0,(phase[j] * math.pi / 180 + k_d * position[j])))  # 指数项体现相位：自身相位和理想相位之间的偏差
            data_array[i+1].append(10*math.log10(abs(a)))  # 作归一化处理，得到标准方向图数值
            data_new.append(data_array[i+1][1])  # 计算得到的方向图

        writer.writerows(data_array)  # 多行写入：将得到的总的data_array的每一个列表写为一行

        plt.plot(data_x, data_new,"y")  #画出现在计算得到的方向图
        plt.show()  # 显示图像


if __name__ == '__main__':
    pattern = Pattern()  # 定义类
    pattern.radiation()  # 调用方向图函数

