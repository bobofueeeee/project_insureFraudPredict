if __name__ == '__main__':
    print('-------开始------')
    from numpy import *
    import  numpy as np



    test = eye(4) # eye(4) 生成对角矩阵
    print(test)

    a = np.array([1,2,3])
    print(a)

    # 最小维度
    a = np.array([1, 2, 3, 4, 5], ndmin=2)
    print(a)

    dt = np.dtype(np.int32)
    print(dt)

    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(-1, 1, 50)
    y = 2 * x + 1
    plt.plot(x, y)
    plt.show()