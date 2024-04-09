# 掌握GA算法实现的主要步骤，以求二元函数的最大值为例
# 个体为某问题的一个解，一组可能解的集合就是种群。寻找种群里适应度最好的个体。
# 对个体进行二进制编码得到染色体，这是我们实际操纵的对象。对于实数编码，DNA长度越长，实数精度越高。
import pandas as pd
import math
import numpy as np

# 中心工作频率和波长
c = 3 * 10 ** 8  # 光速
f1 = 11.5 * 10 ** 9 # 工作中心频率 12.5GHz
k1 = 2 * math.pi * f1 / c  # 相移常数 300=3*10^8/1*10^6 对频率作了处理
f2 = 12.5 * 10 ** 9 # 工作中心频率 12.5GHz
k2 = 2 * math.pi * f2 / c  # 相移常数 300=3*10^8/1*10^6 对频率作了处理
f3 = 13.5 * 10 ** 9 # 工作中心频率 12.5GHz
k3 = 2 * math.pi * f3 / c  # 相移常数 300=3*10^8/1*10^6 对频率作了处理

#  阵元参数
G = 11.5 * 10 ** -3  # 间隔11.5mm
phi0 = 0  # 方位角方向
row = 8 # 阵元行数
column = 8  # 阵元列数
refer_phase = 0 # 参考相位

# 参数初始化
NUM_OF_CELL = 8 # 天线阵元个数(要优化的阵元个数）
DNA_SIZE = NUM_OF_CELL #一条DNA上的的碱基数（二进制编码长度
POP_SIZE = 300 # 种群中总个体数
CROSSOVER_RATE = 0.7 # 交叉率
MUTATION_RATE = 0.2 # 变异率
N_GENERATIONS = 20 # 繁殖总代数（迭代总次数）


prephase = [[90], [0], [90], [0], [0], [0], [90], [0], [90], [90], [90], [0], [0], [0], [0], [90], [0], [90], [0],
            [0], [0], [90], [0], [90], [90], [90], [90], [0], [0], [90], [0], [90], [0], [0], [0], [90], [90], [0],
            [0], [90], [90], [90], [90], [0], [90], [90], [0], [90], [0], [90], [0], [0], [90], [0], [90], [0], [0],
            [0], [90], [90], [90], [90], [90], [0]]
compphase = [[0], [0], [0], [0], [0], [0], [0], [0], [180], [180], [180], [180], [180], [180], [180], [180], [180], [0], [180], [180], [180], [0], [180], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [180], [180], [0], [0], [180], [180], [180], [180], [180], [180], [180], [180], [180], [180], [0], [180], [180], [0], [180], [0], [180], [0], [0], [0], [0], [0], [0], [0], [0]]
# 适应度和选择：保存优秀个体，淘汰不适应环境的个体
def get_fitness(pop,data_x,contributor,n_data): # 计算种群中各个个体的适应度
    x = np.array(translateDNA(pop)) # 调用解码器，x是一个POP_SIZE*NUM_OF_CELL的相位矩阵
    result=[]
    theta = 30  # 仰角方向。共同构成主波束方向
    for i in range(0,POP_SIZE):
        phase = []
        sym_phase = x[:, i:i + 1] # 对角线上的相位个体
        for a in range(0,row):
            for b in range(0,column):
                idx = a * row + b
                if a==b:
                    compphase[idx][0] = sym_phase[a][0]  #用sym_phase中的第i个元素去替换compphase第i行i列的元素
                else:
                    continue
        for j in range(0,row*column):
             phase.append([prephase[j][0]+compphase[j][0]])
        # print("第", i, "种个体最终补偿：", compphase)
        # print("第", i, "种个体最终：", phase)
        r = radiation(data_x, contributor, n_data, phase, k3, theta) #k为相应频率
        result.append(r)
    print("适应度序列为:",result)#  预测值为将解码获得的x,y带入目标函数的结果,这边输入的个数要和NUM_OF_CELL一致，返回一个POP_SIZE长度的
    return result-np.min(result) + 1e-3 # 这边没做归一化，副瓣小适应度大

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
                temp.append(180)
        x.append(temp)
    return x

# 通过选择得到了还不错的基因，但未必是最好的基因。通过繁殖，可能获得更好的基因组合。
# 交叉：子代个体的DNA获得一半父亲的DNA、一半母亲的DNA。算法中，一半由交配点确定，可以是染色体的任意位置。交叉概率一般0.6-1（保证子代有一部分和当前水平一样）
# 变异：DNA既不来自父亲，也不来自母亲，一般为改变一个二进制位。通常是0.1或更小，为了跳出局部最优解。
# 交叉
def crossover_and_mutation(pop, CROSSOVER_RATE):
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
def mutation(child, MUTATION_RATE):
    if np.random.rand() < MUTATION_RATE:
        mutata_point = np.random.randint(0, DNA_SIZE)  # 确定变异点位置
        child[mutata_point] = child[mutata_point] ^ 1 # 和1做异或运算，即反转
# 选择：适应度越高，被选择的机会越大，而非一定被选择，从而避免早熟
def select(pop, fitness): # 根据适应度进行选择
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=(fitness)/(fitness.sum()))  # 轮盘赌思想
    # print("原种群为:", pop, "处理后的适应度为：", fitness, "没被淘汰的个体序号为:", idx, "新种群:", pop[idx])
    # 主要使用choice参数p功能，描述了从np.arange()中选择每一个元素的的概率
    return pop[idx]  #返回一个索引列表，完成筛选

# 针对打印最优的处理
def translate_single(pop): # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    phase=[]
    for a in range(0, row):
        for b in range(0, column):
            idx = a * row + b
            if a == b:
                compphase[idx][0] = pop[a]*180  # 用sym_phase中的第i个元素去替换compphase第i行i列的元素
            else:
                continue
    for j in range(0, row * column):
        phase.append([prephase[j][0] + compphase[j][0]])
    return phase

# 遗传算法
def evolution():
    # 生成二维数组，行数为POP_SIZE，列数为DNA_SIZE
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # 个体数为POP_SIZE，每个个体的二进制编码量为POP_SIZE
    data_x, data_y, n_data = read_csv()  # 获取单个阵元的数据
    contributor=[]
    for i in range(0, n_data):  # 分贝值和实际值之间的换算，y是归一化的
        contributor.append(10**(data_y[i]/10))
    for i in range(0,N_GENERATIONS): # 迭代N代
        fitness = get_fitness(pop,data_x,contributor,n_data)  # 先计算父代的适应度
        print("第", i+1, "次迭代")
        pop = select(pop, fitness)  # 父代中适应度低的被淘汰，无法参与繁衍
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))  #交叉变异得到子代
        # print("子代为：", pop)
    print(pop)  #看最后一次运行结果
    fitness=get_fitness(pop,data_x,contributor,n_data)
    max_fitness_index = np.argmax(fitness)
    bestfinalphase = translate_single(pop[max_fitness_index])
    bestcompphase = []
    for i in range(row*column):
        bestcompphase.append([bestfinalphase[i][0]-prephase[i][0]])
    print("max_fitness_index:",max_fitness_index)
    print("max_fitness:", fitness[max_fitness_index])  # 获取适应度值
    print("best DNA:", pop[max_fitness_index])
    print("best answer:", bestcompphase)

# 读取单个阵元的信息
def read_csv():
    data_x = []  # 初始化空x、y列表，初始时不采用归一化（负数处理需要转弯）
    data_y = []  # 归一化
    csv_data = pd.read_csv("../dataset/Gain_for_11.5GHz.csv") # 同文件夹下的data文件，读取单个阵元数据
    n_data = len(csv_data)  # 定义数据长度（循环总长）为从文件中获得数据条数（函数认为表头不占行）
    # 将表格文件转为两个有映射关系的一维向量：第0列为，第1列为
    for i in range(0, n_data):  # 0循环到n_data-1，共n_data次
        data_x.append(csv_data.loc[i][2])  # x列表中插入文件第i行第0列的数据：扫描角度（精度为1°）
        data_y.append(csv_data.loc[i][3])  # 归一
    return data_x,data_y, n_data

# 初始化坐标和激励幅值矩阵
def parameter_initial():  # 基本初始数据计算和获取
    position = []  # 点源位置
    amplititude = []  # 激励幅值
    # zeta = []
    for i in range(0, row):  # 这边获得的已经是8*8
        for j in range(0, column):
            position.append([i*G, j*G])  # 最终获得一个
            amplititude.append(1)
            # zeta.append(360*G*f2/c*(i*np.cos(phi0)+j*np.sin(phi0))*np.sin(theta0*np.pi/180))
    return position, amplititude

# 找副瓣
def get_sll(gain):
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
    # print(psb_idx, psb_num) #满足条件的所有数
    max_idx = psb_num.index(max(psb_num))
    # print('最大值为', max(psb_num))
    psb_num.pop(max_idx)
    psb_idx.pop(max_idx)  #删去对最大值的记忆
    second_idx = psb_num.index(max(psb_num))  # 剩下的最大即为副瓣
    return second_idx, arr[psb_idx[second_idx]]

# 计算某个矩阵条件下的方向图并返回相应副瓣值
def radiation(data_x,contributor, n_data, phase, k, theta0):
    position,  amplititude = parameter_initial()  # 根据初始给定值获得坐标和激励矩阵
    data_array = []  # 初始化数据列表
    data_new = []  # 初始化new列表

    for i in range(0, n_data):  # 生成数据长度和hfss得到的csv文件长度保持一致（才能做对比）
        a = complex(0, 0)  # 初始化a为复数0+j0。对所扫描的某个角度。每个都需要重新初始化a
        k_d = k * np.sin(data_x[i] * np.pi / 180)
        for j in range(0,row*column):
            a = a +  amplititude[j] * contributor[i] * np.exp(complex(0, (phase[j][0] * np.pi / 180 + + k_d*(position[j][0] * np.cos(phi0) + position[j][1] * np.sin(phi0)))))
        data_new.append(abs(a))  # 不作归一化处理，得到标准方向图数值
    max_data = np.max(data_new)
    for i in range(0, n_data):
        gain_normal = 20 * math.log10(data_new[i] / max_data)  # 功率和电场的平方关系对数化后为二倍关系
        data_array.append(gain_normal)  # 做归一化（不如一开始就归一化）

    second_idx, second_value = get_sll(data_array)  # 查找副瓣

    return data_array[180+theta0]-second_value

if __name__ == "__main__":
    evolution()




