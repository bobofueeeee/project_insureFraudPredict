if __name__ == '__main__':

    import matplotlib.pyplot as plt
    #%matplotlib inline

    def cumulate(data, col):
        print('缺失值占比:' + str(data[col].isnull().sum() / data.shape[0]))

        print('类别有' + str(data[col].value_counts().shape) + '个')
        print('类别特征情况' + '\n' + str(data[col].value_counts()))
        tp = (data[col].value_counts() / data.shape[0]).values.flatten().tolist()
        cumulate = []
        for i in range(1, len(tp) + 1):
            cumulate.append(sum(tp[0:i]))
        plt.plot(cumulate)

    print("---------开始数据分析---------")
    import pandas as pd


    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)  ## 显示全部结果，不带省略点
    pd.set_option('display.width',1000)
    pd.set_option('display.float_format', '{:.0f}'.format)

    test_identity = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\test_identity.csv")
    train_identity = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\train_identity.csv")

    for analysis in [test_identity, train_identity]:
        print('----------------analysis----------------')
        print(analysis.shape)
        i = 5
        analysis = analysis.head(i)
        analysis.insert(0, 'new_col', '真实数据')  ## 在第一列，加入列（new_col2），并赋值为 “真实数据”
        analysis = analysis.append(test_identity.dtypes, ignore_index=True)
        analysis['new_col'].loc[int(i)] = '数据类型'
        analysis = analysis.append(test_identity.isnull().sum(), ignore_index=True)
        analysis['new_col'].loc[int(i + 1)] = '空值数量'
        analysis = analysis.append(test_identity.min(), ignore_index=True)
        analysis['new_col'].loc[int(i + 2)] = '最小值'
        analysis = analysis.append(test_identity.max(), ignore_index=True)
        analysis['new_col'].loc[int(i + 3)] = '最大值'
        analysis = analysis.append(test_identity.describe().loc['mean'], ignore_index=True)
        analysis['new_col'].loc[int(i + 4)] = '平均值'
        analysis = analysis.append(test_identity.describe().loc['std'], ignore_index=True)
        analysis['new_col'].loc[int(i + 5)] = '方差'

        analysis_part02 = analysis.head(i)
        # analysis = analysis.drop([0,1,2]) # 删除指定列
        analysis.drop(analysis.index[0:i], inplace=True)  # 删除切片行
        analysis = analysis.append(analysis_part02, ignore_index=True)

        # print(test_identity.describe())
        print(analysis)


    # i = 1
    # while i < 39:
    #     id = 'id_' + ("0%s" % i)
    #     print(id)
    #     cumulate(train_identity,id)
    #     plt.hist(train_identity[id].dropna()) # train_identity.id_01 = train_identity['id_01']
    #     i = i + 1
    plt.show()
    cumulate(train_identity, 'DeviceInfo')
    plt.hist(train_identity.DeviceInfo.dropna(),bins=50)

    print('结束')


