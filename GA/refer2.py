# -*-coding:utf-8 -*-
#目标：求解2*sin(x)+cos(x)最大值

import random
import math
import matplotlib.pyplot as plt

# 种群初始化
# 初始化生成chromosome_length大小的population_size个个体得到二进制基因型种群
def species_origin(population_size, chromosome_length):
    population=[[]]
    for i in range(population_size):
        temporary=[] # 染色体暂存器
        for j in range(chromosome_length):  # 总共的染色体数
            temporary.append(random.randint(0,1))  # 随机产生一个染色体，由二进制数组成
        population.append(temporary)  # 将个体（染色体序列添加到种群中）
    return population[1:]  # 将种群（一个个体和染色体两维数组）返回，从1切片获得，可能和range有关

# 编码：将二进制的染色体基因编码成十进制的表现型
def translation(population, chromosome_length):
    temporary=[]
    for i in range(len(population)):
        total = 0  #每计算一个个体都要重新初始化一次
        for j in range(chromosome_length):
            total+=population[i][j]*(math.pow(2,j))  # 从第一个个体开始，逐个基因对2求j次幂，再求和
        temporary.append(total)  # 二进制到十进制转换完成
    return temporary  # 返回种群中所有个体编码完成后对应的十进制数（组）

def b2d(b, max_value, chromosome_length):
    total = 0
    for i in range(len(b)):
        total = total + b[i]*math.pow(2, i)  # 从第一位开始，每一位对2求幂，求和
    total = total*max_value/(math.pow(2, chromosome_length)-1)
    return total

# 适应度计算：函数值总取非负值，以最大化为优化目标，目标函数2*sin(x)+cos(x)即环境
def function(population, chromosome_length, max_value):
    temporary = []
    function1 = []
    temporary = translation(population, chromosome_length)
    # 暂存种群中的所有染色体（十进制）
    for i in range(len(temporary)):
        x = temporary[i]*max_value/(math.pow(2, chromosome_length)-1)  # 线性变换，建立定义域和二进制序列得到变换关系
        # 不同的映射关系会影响收敛速度和平滑性
        function1.append(2*math.sin(x) + math.cos(x))
    return function1

# 只保留非负值得到适应度
def fitness(function1):
    fitness1 = []
    min_fitness = mf =0
    for i in range(len(function1)): # 如果适应度小于0，则定为0
        if(function1[i] + mf > 0):
            temporary = mf + function1[i]
        else:
            temporary = 0.0
    fitness1.append(temporary)  # 将适应度添加到表中

    return fitness1

def best(population, fitness1):
    px = len(population)
    bestindividual = []
    bestfitness = fitness1[0]

    for i in range(1,px): # 遍历找到最大的适应度
        if(fitness1[i] > bestfitness):
            bestfitness = fitness1[i]
            bestindividual = population[i]

    return [bestindividual, bestfitness]

# 选择:轮盘赌算法
# 计算出所有个体的适应度总和
def sum(fitness1):
    total = 0
    for i in range(len(fitness1)):
        total+= fitness[i]
    return total

# 计算适应度斐波那契列表，即求出累计适应度
def cumsum(fitness1):
    for i in range(len(fitness1)-2, -1, -1):  # 逆序，从fitness-2到0？ range(start,stop,[step])
        total = 0
        j = 0
        while(j<=i):  # 将适应度划分成区间
            total+=fitness[j]
            j+=1
        fitness[i]=total
        fitness[len(fitness1)-1]=1

# 产生一个0到1之间得到随机数，依据随机数出现在斐波那契列表哪个区间来确定各个个体被选中的次数
def selection(population, fitness1):
    new_fitness = []  # 适应性暂存器
    total_fitness = sum(fitness1)  # 所有适应度求和
    for i in range(len(fitness1)):  # 所有适应度概率化
        new_fitness.append(fitness[i]/total_fitness)
    cumsum(new_fitness)  # 将所有适应度划分成区间
    ms = []  # 存活的种群
    population_length = pop_len = len(population) # 求出当下种群长度
    for i in range(pop_len): # 根据生成的随机数确定哪几个能存活
        ms.append(random.random()) # 产生随机值
        ms.sort() # 存活的种群排序
        fitin = 0
        newin = 0
        new_population = new_pop = population

    # 轮盘赌方式
    while newin<pop_len:
        if(ms[newin]<new_fitness[fitin]):
            new_pop[newin]=population[fitin]
            newin+=1
        else:
            fitin+=1
    population=new_pop

# 交叉
def crossover(population, pc): # pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体
    pop_len = len(population)

    for i in range(pop_len-1):
        cpoint = random.randint(0, len(population[0]))  # 在种群个数内随机生成单点交叉点
        temporary1 = []
        temporary2 = []

        temporary1.extend(population[i][0:cpoint])  # 将temp1作为存放第i条染色体0到cpoint个基因的暂存器
        temporary1.extend(population[i+1][cpoint:len(population[i])])  # 把temp2作为存放第i+1个染色体cpoint到最后的基因

        temporary2.extend(population[i+1][0:cpoint])  # 将temp2存放第i+1个染色体中前0到cpoint个基因
        temporary2.extend(population[i+1][cpoint:len(population[i])])  # temp2存放第i个染色体cpoint后的基因

        population[i] = temporary1  # 交叉完成
        population[i+1] = temporary2

# 变异
def mutation(population, pm):  # pm是概率阈值
    px = len(population) # 种群中所有种群个体的个数
    py = len(population[0])  # 个体中基因的个数
    for i in range(px):
        if(random.random() < pm):  # 如果生成的随机数小于阈值就变异
            mpoint = random.randint(0, py-1) # 生成[0,py-1]的随机数
            if(population[i][mpoint]==1):  # 将mpoint处的基因进行单点随机变异
                population[i][mpoint] = 0
            else:
                population[i][mpoint] = 1


if __name__ == "__main__":
    population_size = 500
    max_value = 10
    # 基因中允许出现的最大值
    chromosome_length = 10
    pc = 0.6
    pm = 0.01

    results = [[]]
    fitness1 = []
    fitmean = []

    population = pop = species_origin(population_size, chromosome_length)
    # 生成一个初始的种群

    for i in range(population_size):  # 注意这里是迭代500次
        function1 = function(population, chromosome_length, max_value)
        fitness1 = fitness(function1)
        best_individual, best_fitness = best(population, fitness1)
        results.append([best_fitness, b2d(best_individual, max_value, chromosome_length)])
        # 将最好的个体和最好的适应度保存，并将最好的个体转成十进制
        selection(population, fitness1)  # 选择
        crossover(population, pc)  # 交配
        mutation(population, pm)  # 变异

    results = results[1:]
    results.sort()
    X = []
    Y = []
    for i in range(500):  # 500轮的结果
        X.append(i)
        Y.append(results[i][0])
    plt.plot(X, Y)
    plt.show()