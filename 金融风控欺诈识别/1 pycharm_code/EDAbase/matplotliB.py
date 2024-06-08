
if __name__ == '__main__':


    import matplotlib.pyplot as plt
    import numpy as np
    print('-------开始------')
    x = np.linspace(-1, 1, 50)
    y = 2 * x + 1
    plt.plot(x, y)
    plt.show()