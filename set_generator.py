import pandas as pd
import math
import cmath
import csv
import numpy as np
import matplotlib.pyplot as plt

prephase = [[0], [0], [90], [90], [0], [0], [90], [90], [0], [90], [90], [0], [90], [0], [0], [90], [0], [0], [90], [0], [90], [90], [90], [0], [0], [0], [0], [0], [90], [90], [90], [0], [90], [90], [0], [0], [90], [90], [0], [90], [90], [90], [0], [0], [0], [0], [90], [90], [90], [0], [90], [0], [0], [0], [0], [90], [90], [0], [90], [0], [90], [0], [90], [90]]
len_prephase=len(prephase)
row=str(int(math.sqrt(len_prephase)))
column=row
str1="hello hudson"

with open("example.txt",'w') as file:
    file.write(str1)
    file.writelines('\n'+row+'_'+column)
    file.close()