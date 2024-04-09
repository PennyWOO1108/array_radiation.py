import numpy as np

if __name__ == "__main__":
    prephase = [[90], [0], [90], [0], [0], [0], [90], [0], [90], [90], [90], [0], [0], [0], [0], [90], [0], [90], [0],
                [0], [0], [90], [0], [90], [90], [90], [90], [0], [0], [90], [0], [90], [0], [0], [0], [90], [90], [0],
                [0], [90], [90], [90], [90], [0], [90], [90], [0], [90], [0], [90], [0], [0], [90], [0], [90], [0], [0],
                [0], [90], [90], [90], [90], [90], [0]]
    subphase = np.zeros([64,1])
    finalphase = []
    for i in range(64):
        finalphase.append(prephase[i][0]-subphase[i][0])
    print(finalphase)