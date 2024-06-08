# coding=utf-8
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    #%matplotlib inline

    ## dataframe优化
    def reduce_mem_usage(df):
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.
        """
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df

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
    import modin.pandas as pd


    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)  ## 显示全部结果，不带省略点
    pd.set_option('display.width',1000)
    pd.set_option('display.float_format', '{:.0f}'.format)
    pd.set_option("modein.engine","Dask")

    test_identity = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\test_identity.csv")
    # test_identity = reduce_mem_usage(test_identity)
    train_identity = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\train_identity.csv")
    # train_identity = reduce_mem_usage(train_identity)


    for analysis in [test_identity,train_identity]:
        print('----------------analysis----------------')
        analysis_base = analysis
        print(analysis_base.shape)
        i = 5
        analysis = analysis_base.head(i)
        analysis.insert(0, 'new_col', '真实数据')  ## 在第一列，加入列（new_col2），并赋值为 “真实数据”
        analysis = analysis.append(analysis_base.dtypes, ignore_index=True)
        analysis['new_col'].loc[int(i)] = '数据类型'
        analysis = analysis.append(analysis_base.isnull().sum(), ignore_index=True)
        analysis['new_col'].loc[int(i + 1)] = '空值数量'
        analysis = analysis.append(analysis_base.min(), ignore_index=True)
        analysis['new_col'].loc[int(i + 2)] = '最小值'
        analysis = analysis.append(analysis_base.max(), ignore_index=True)
        analysis['new_col'].loc[int(i + 3)] = '最大值'
        analysis = analysis.append(analysis_base.describe().loc['mean'], ignore_index=True)
        analysis['new_col'].loc[int(i + 4)] = '平均值'
        analysis = analysis.append(analysis_base.describe().loc['std'], ignore_index=True)
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
        cumulate(test_identity, 'id_01')
        plt.hist(test_identity.id_01.dropna(),bins=50)


    print('结束')


