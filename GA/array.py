#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from aimfunc import aimfunc  # 导入 自定义 的目标函数
import time

"""============================变量设置============================"""
phase1 = [0, 359]  # 第一个相位变量范围
phase2 = [0, 359]  # 第二个相位变量范围
phase3 = [0, 359]
phase4 = [0, 359]
phase5 = [0, 359]
phase6 = [0, 359]

b1 = [1, 1]  # 第一个相位变量边界，1表示包含范围的边界，0表示不包含
b2 = [1, 1]  # 第二个相位变量边界，1表示包含范围的边界，0表示不包含
b3 = [1, 1]
b4 = [1, 1]
b5 = [1, 1]
b6 = [1, 1]

ranges = np.vstack([phase1, phase2, phase3, phase4, phase5, phase6]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
borders = np.vstack([b1, b2, b3, b4, b5, b6]).T  # 生成自变量的边界矩阵
varTypes = np.array([0, 0, 0, 0, 0, 0])  # 决策变量的类型，0表示连续，1表示离散
"""==========================染色体编码设置========================="""
Encoding = 'BG'  # 'BG'表示采用二进制/格雷编码
codes = [0, 0, 0, 0, 0, 0]  # 决策变量的编码方式，设置两个0表示两个决策变量均使用二进制编码
precisions = [1, 1, 1, 1, 1, 1]  # 决策变量的编码精度
scales = [0, 0, 0, 0, 0, 0]  # 0表示采用算术刻度，1表示采用对数刻度
FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)  # 调用函数创建译码矩阵
"""=========================遗传算法参数设置========================"""
NIND = 4  # 种群个体数目
MAXGEN = 2  # 最大遗传代数
maxormins = [1]  # 列表元素为1则表示对应的目标函数是最小化，元素为-1则表示对应的目标函数是最大化
selectStyle = 'rws'  # 采用轮盘赌选择
recStyle = 'xovdp'  # 采用两点交叉
mutStyle = 'mutbin'  # 采用二进制染色体的变异算子
pc = 0.7  # 交叉概率
pm = 1  # 整条染色体的变异概率（每一位的变异概率=pm/染色体长度）
Lind = int(np.sum(FieldD[0, :]))  # 计算染色体长度
obj_trace = np.zeros((MAXGEN, 2))  # 定义目标函数值记录器
var_trace = np.zeros((MAXGEN, Lind))  # 染色体记录器，记录历代最优个体的染色体
"""=========================开始遗传算法进化========================"""
start_time = time.time()  # 开始计时
Chrom = ea.crtpc(Encoding, NIND, FieldD)  # 生成种群染色体矩阵
variable = ea.bs2real(Chrom, FieldD)  # 对初始种群进行解码
print(variable)

ObjV = aimfunc(variable)  # 计算初始种群个体的目标函数值
FitnV = ea.ranking(maxormins * ObjV)  # 根据目标函数大小分配适应度值
best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
# 开始进化
for gen in range(MAXGEN):
    SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND - 1), :]  # 选择
    SelCh = ea.recombin(recStyle, SelCh, pc)  # 重组
    SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)  # 变异
    # 把父代精英个体与子代的染色体进行合并，得到新一代种群
    Chrom = np.vstack([Chrom[best_ind, :], SelCh])
    Phen = ea.bs2real(Chrom, FieldD)  # 对种群进行解码(二进制转十进制)
    ObjV = aimfunc(Phen)  # 求种群个体的目标函数值
    FitnV = ea.ranking(maxormins * ObjV)  # 根据目标函数大小分配适应度值
    # 记录
    best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
    obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]  # 记录当代种群的目标函数均值
    obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
    var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体
# 进化完成
end_time = time.time()  # 结束计时
ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])  # 绘制图像
"""============================输出结果============================"""
best_gen = np.argmax(obj_trace[:, [1]])
print('最优解的目标函数值：', obj_trace[best_gen, 1])
variable = ea.bs2real(var_trace[[best_gen], :], FieldD)  # 解码得到表现型（即对应的决策变量值）
print('最优解的决策变量值为：')
for i in range(variable.shape[1]):
    print('x' + str(i) + '=', variable[0, i])
print('用时：', end_time - start_time, '秒')
