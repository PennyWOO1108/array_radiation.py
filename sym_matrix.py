import numpy as np
import random

# value = [8, 10, 5, 10, 13, 11, 2, 13, 9, 12, 19, 6, 2, 14, 11, 10,11, 9, 12, 9]  # 表征每行为0的个数

def descend_sort(x, N):
    descend_x = []  # x降序排列结果
    descend_idx = []  # 各数对应的原索引
    for i in range(N):
        descend_x.append(max(x))
        descend_idx.append(x.argmax())
        x[x.argmax()] = 0
    return descend_x, descend_idx

def get_matrix():
    value = [4, 3, 3, 2, 2]  # GA算法计算得到的每行应有多少个0的向量
    x = np.array(value)
    N = 5
    for i in range(0,N): #对行遍历，沿对角线方向走。
        sample = random.sample(range(i,N),x[i])  # 根据每列所需0数随机选取x[i]个位置置零
        print("产生样本:", len(sample),sample)
        for j in range(len(sample)):
            if sample[j] == i:  # 对角线位置
                phase_matrix[i][sample[j]] = 0 # 置零
                x[sample[j]]=x[sample[j]]-1
            elif sample[j] != i:  # 非对角线位置
                phase_matrix[i][sample[j]] = 0 # 第descend_idx[i]行第sample(j)列置为0
                phase_matrix[sample[j]][i] = 0 # 第sample(j)行第descend_idx[i]列置为0
                x[i] = x[i]-1
                x[sample[j]] = x[sample[j]]-1
    print(phase_matrix)

def get_descend_matrix():
    value = [3,2,1,4,3]
    x = np.array(value)
    N = 6
    phase_matrix = np.ones((N, N))



if __name__ == '__main__':
    value = [3, 2, 1, 2, 3, 4]
    x = np.array(value)
    N = 6
    phase_matrix = np.ones((N,N))
    print(max(x), x.argmax())
    descend_x, descend_idx = descend_sort(x, N)
    print("降序排列为：", descend_x, "对应索引为：", descend_idx)
    get_matrix()

