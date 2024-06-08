import pandas as pd
import numpy as np




list = [[[1,2,3],[1,2,3],[1,2,3]],[1],['xxxx']]
print('-----------list--------------')
print(list)

df = pd.DataFrame(list)
print('-----------DataFrame--------------')
print(df)

s = pd.Series(list)
print('-----------Series--------------')
print(s)

a = np.array(list)
print('-----------numpy--------------')
print(a)

print('-----------numpy一维数组--------------')
a2 = np.array([1,2,3])
print(a2)

print('-----------numpy二维数组--------------')
a3 = np.array([[1,2,3],['x','b']])
print(a3)