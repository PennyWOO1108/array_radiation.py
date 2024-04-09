# 掌握GA算法实现的主要步骤，以求二元函数的最大值为例
# 个体为某问题的一个解，一组可能解的集合就是种群。寻找种群里适应度最好的个体。
# 对个体进行二进制编码得到染色体，这是我们实际操纵的对象。对于实数编码，DNA长度越长，实数精度越高。
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math
import cmath
import csv
import numpy as np
import matplotlib.pyplot as plt


# 中心工作频率和波长
c = 3 * 10 ** 8  # 光速
fc = 12.5 * 10 ** 9 # 工作中心频率 12.5GHz
fl = 12 * 10 ** 9
fh = 13 * 10 ** 9
# lamb = c / f

kc = 2 * math.pi * fc / c   # 相移常数 300=3*10^8/1*10^6 对频率作了处理
kl = 2 * math.pi * fl / c
kh = 2 * math.pi * fh / c

#  阵元参数
G = 11.5 * 10 ** -3  # 间隔11.5mm
phi = 0  # 方位角方向
theta0 = 0  # 仰角方向。共同构成主波束方向
row = 4  # 阵元行数
column = 4  # 阵元列数

# 参数初始化
NUM_OF_CELL = row*column # 天线阵元个数
DNA_SIZE = NUM_OF_CELL #一条DNA上的的碱基数（二进制编码长度
POP_SIZE = 100 # 种群中总个体数
CROSSOVER_RATE = 0.8 # 交叉率
MUTATION_RATE = 0.1 # 变异率
N_GENERATIONS = 5 # 繁殖总代数（迭代总次数）

# 目标函数：被替换为找最小
# def F(x,data_x, data_y, n_data):
#     result = radiation(data_x, data_y, n_data, x)
#     return result
# def F(x):
#     result = []
#     for i in range(POP_SIZE):
#         temp = 0
#         for j in range(NUM_OF_CELL):
#             # print(x[j][i])
#             temp += x[j][i]
#         # print("第", i, "个个体的函数值为:", temp)
#         result.append(temp)
#     return result

# 绘图模块
# def plot_3d(ax):
# 	X = np.linspace(*X_BOUND, 100)
# 	Y = np.linspace(*Y_BOUND, 100)
# 	X,Y = np.meshgrid(X, Y)
# 	Z = F(X, Y)
# 	ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm)
# 	ax.set_zlim(-10,10)
# 	ax.set_xlabel('x')
# 	ax.set_ylabel('y')
# 	ax.set_zlabel('z')
# 	plt.pause(3)
# 	plt.show()
# 适应度和选择：保存优秀个体，淘汰不适应环境的个体
#
def get_fitness(pop): # 计算种群中各个个体的适应度
    data_x, data_y, n_data = read_csv()  # 获取单个阵元的数据
    x = np.array(translateDNA(pop)) # 调用解码器，x是一个POP_SIZE*NUM_OF_CELL的相位矩阵
    weight=[0.5,0.25,0.25]
    # print("解码结果为：",x, "矩阵尺寸为：", len(x))
    result1= radiation(data_x, data_y, n_data,x,kc) #  预测值为将解码获得的x,y带入目标函数的结果,这边输入的个数要和NUM_OF_CELL一致
    print("结果是：",result1)
    # result2 = radiation(data_x, data_y, n_data,x,kl)
    # result3 = radiation(data_x, data_y, n_data, x, kh)
    # result = np.multiply(weight[0],result1)+np.multiply(weight[1],result2)+np.multiply(weight[2],result3)
    # return (np.max(result1)+result1 + 1e-3) ** 2 # 这边没做归一化，函数值大适应度大
    return (np.max(result1)+result1) # 这边没做归一化，函数值大适应度大

# 解码  what is pop
def translateDNA(pop): # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    X = []
    for i in range(NUM_OF_CELL):
        if i < NUM_OF_CELL - 1:
            X.append(pop[:,i:i+1])
        else:
            X.append(pop[:, i:])
    x = []
    for i in range(NUM_OF_CELL):  #遍历X[1]X[2]X[3]切片组
        temp = []
        for j in range(POP_SIZE):
            if X[i][j] != 1:
                temp.append(0)
            elif X[i][j] == 1:
                temp.append(90)
        x.append(temp)
    return x
# 通过选择得到了还不错的基因，但未必是最好的基因。通过繁殖，可能获得更好的基因组合。
# 交叉：子代个体的DNA获得一半父亲的DNA、一半母亲的DNA。算法中，一半由交配点确定，可以是染色体的任意位置。交叉概率一般0.6-1（保证子代有一部分和当前水平一样）
# 变异：DNA既不来自父亲，也不来自母亲，一般为改变一个二进制位。通常是0.1或更小，为了跳出局部最优解。
# 交叉
def crossover_and_mutation(pop, CROSSOVER_RATE = 0.8):
    new_pop = [] # 初始化子代种群
    for father in pop: # 遍历种群中每一个个体并将其作为父亲，可以产生相同数量的子代
        child = father # 子代先得到父本得到全部基因，即相同的二进制序列
        if np.random.rand() < CROSSOVER_RATE:
            # print("发生交叉")
            mother = pop[np.random.randint(POP_SIZE)] # 随机从种群中选取另一个个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE)  #随机产生交叉点
            child[cross_points:] = mother[cross_points:] # 切片替换切片，孩子位于交叉点之后的基因替换为母本的
        mutation(child, MUTATION_RATE) # 每个后代有一定几率变异
        new_pop.append(child) # 成为子代中的一个个体
    return new_pop # 返回子代
# 变异
def mutation(child, MUTATION_RATE=0.1):
    if np.random.rand() < MUTATION_RATE:
        mutata_point = np.random.randint(0, DNA_SIZE)  # 确定变异点位置
        child[mutata_point] = child[mutata_point] ^ 1 # 和1做异或运算，即反转
# 选择：适应度越高，被选择的机会越大，而非一定被选择，从而避免早熟
def select(pop, fitness): # 根据适应度进行选择
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=(fitness)/(fitness.sum()))  # 轮盘赌思想
    # print("原种群为:", pop, "处理后的适应度为：", fitness, "没被淘汰的个体序号为:", idx, "新种群:", pop[idx])
    # 主要使用choice参数p功能，描述了从np.arange()中选择每一个元素的的概率
    return pop[idx]  #返回一个索引列表，完成筛选
# 打印最优个体信息
def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)  # 找到适应度最大个体的索引
    print("max_fitness:", fitness[max_fitness_index])  # 获取适应度值
    # print("best DNA:", pop[max_fitness_index])
    print("best answer:", translate_single(pop[max_fitness_index]))
# 针对打印最优的处理
def translate_single(pop): # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    # print(pop)
    X = []
    for i in range(NUM_OF_CELL):
        if i < NUM_OF_CELL - 1:
            X.append(pop[i:i+1])
        else:
            X.append(pop[i:])
    # print(X)
    x = []
    for i in range(NUM_OF_CELL):
        temp = []
        if X[i][0] != 1:
            temp.append(0)
        elif X[i][0] == 1:
            temp.append(90)
        x.append(temp)

    return x
# 遗传算法
def evolution():
    # 生成二维数组，行数为POP_SIZE，列数为DNA_SIZE
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # 个体数为POP_SIZE，每个个体的二进制编码量为POP_SIZE
    for i in range(N_GENERATIONS): # 迭代N代
        print("第", i, "次迭代")
        fitness = get_fitness(pop)  # 先计算父代的适应度
        pop = select(pop, fitness)  # 父代中适应度低的被淘汰，无法参与繁衍
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))  #交叉变异得到子代
        # print("子代为：", pop)
        print_info(pop)  # 打印当代最好个体的信息

# 读取单个阵元的信息
def read_csv():
    data_x = []  # 初始化空x、y列表
    data_y = []  # 没归一化
    csv_data = pd.read_csv("../dataset/Gain2.csv") # 同文件夹下的data文件，读取单个阵元数据
    n_data = len(csv_data)  # 定义数据长度（循环总长）为从文件中获得数据条数
    for i in range(0, n_data):  # 0循环到n_data-1，共n_data次
        data_x.append(csv_data.loc[i][0])  # x列表中插入文件第i行第0列的数据：扫描角度
        data_y.append(csv_data.loc[i][1])  # 没归一化
    return data_x,data_y, n_data
# 初始化坐标和激励幅值矩阵
def parameter_initial():  # 基本初始数据计算和获取

    position = []  # 点源位置
    for i in range(0, row):  # 这边获得的已经是8*8
        for j in range(0, column):
            position.append([i*G, j*G])

    return position
# 找副瓣
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
    # print(psb_idx, psb_num)
    max_idx = psb_num.index(max(psb_num))
    # print('最大值为', max(psb_num))
    psb_num.pop(max_idx)
    psb_idx.pop(max_idx)
    second_idx = psb_num.index(max(psb_num))
    # print('副瓣值为：', arr[psb_idx[second_idx]])
    return second_idx, arr[psb_idx[second_idx]]
# 计算某个矩阵条件下的方向图并返回相应副瓣值
def radiation(data_x, data_y, n_data, phase,k):
    position = parameter_initial()  # 根据初始给定值获得坐标和激励矩阵
    data_array = []  # 初始化数据列表
    contributor = []
    # gain = []
    # second_value = []
    performance = []
    max_angle =int((n_data-1)/2 + theta0/2 ) # 寻找副瓣低、增益大

    for i in range(0, n_data):
        contributor.append(10**(data_y[i]/10))  # 分贝值和实际值之间的换算，y2是没归一化的

    for x in range(0,POP_SIZE):
        for i in range(0, n_data):  # 生成数据长度和hfss得到的csv文件长度保持一致（才能做对比）
            a = complex(0, 0)  # 初始化a为复数0+j0。对所扫描的某个角度。每个都需要重新初始化a
            k_d = k * (np.sin(data_x[i] * np.pi / 180) - np.sin(theta0 * np.pi / 180))  # 1*n_data的向量
            for j in range(0,row*column):
                zeta = k_d * (position[j][0] * np.cos(phi) + position[j][1] * np.sin(phi))
                a = a + contributor[i] * np.exp(complex(0, (phase[j][x] * np.pi / 180 + zeta)))
            data_array.append(10 * math.log10(abs(a)))  # 不作归一化处理，得到标准方向图数值
        # gain.append(data_array[theta0])
        idx, value = get_sll(data_array)
        # second_value.append(value)
        performance.append(data_array[max_angle]-value)

    print(performance)

    return performance

if __name__ == "__main__":
    evolution()
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，继续执行
    # plot_3d(ax)
    #     # if 'sca' in locals():
    #     #     sca.remove()
    #     # sca = ax.scatter(x, y, F(x,y), c='black',marker='o'); plt.show(); plt.pause(0.1)

#     pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))
#     print(pop)
#     x = translateDNA(pop)
#     print(x)



