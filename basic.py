# A system that uses a genetic algorithm to maximize a function of many variables
# ref https://blog.floydhub.com/introduction-to--genetic-algorithms/
import random
import sys
import math

# 适应度函数计算：四个变量，最大值应该在w=-0.25 x=3 y=-1 z=2
def fitness_function(w,x,y,z):
    return -2*(w ** 2) + math.sqrt(abs(w)) - (x ** 2) + (6 * x) - (y ** 2) - (2 * y) - (z ** 2) + (4 * z)
# 简单的适应度函数计算例子：两个变量，最大值在x=1,y=2
def simple_fitness_function(x,y):
    return - (x** 2) + (2 * x) - (y ** 2) + (4 * y)

# 判断采用哪一种计算逻辑（我们可能不需要使用）
def evaluate_generation(population):
    scores = []
    total = 0
    for individual in population:
        if len(individual) == 2:
            r = simple_fitness_function(individual[0], individual[1])
            scores.append(r)
            total += r
        elif len(individual) == 4:
            r = fitness_function(individual[0], individual[1], individual[2], individual[3])
            scores.append(r)
            total += r
        else:
            print("error:Wrong nuber of arguments received")
    avg = total / len(scores)  # 求一个用于选择的平均值

    return scores, avg

# create child from parent
def mutate(individual):  # 变异，不断产生随机数的过程
    new = []
    for attribute in individual:
        new.append(attribute + random.normalvariate(0,attribute + .1)) # 正态分布随机数
    return new

# 在最后的时候找到最优解并输出（感觉这步可以合并进求解里）
def find_best(population):
    best = None
    val = None
    for individual in population:
        if len(individual) == 2:
            r = simple_fitness_function(individual[0], individual[1])
            try:
                if r > val:
                    best = individual
                    val = r
            except:  #On the first run, set the result as the best
                    best = individual
                    val = r
        elif len(individual) == 4:
            r = fitness_function(individual[0], individual[1], individual[2], individual[3])
            try:
                if r > val:
                    best = individual
                    val = r
            except:  # On the first run, set the result as the best
                    best = individual
                    val = r
        else:
            print("error: Wrong number of arguments received")
    return best, val

# 采用随机方法生成初始种群：n为个体数，p为决策变量数
def initialize(n, p):
    pop = [[0] * n]  # 这是个啥
    for i in range(p):  # 这添加了几个数
        pop.append(mutate(pop[0]))
    return pop


