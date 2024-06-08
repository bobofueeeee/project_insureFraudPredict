


if __name__ == '__main__':
    print("---------开始---------")
    import pandas as pd


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width',1000)

    test_identity = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\test_identity.csv")
    test_identity_analysis = test_identity.head(3)

    # 列操作
    test_identity_analysis.insert(0,'new_col','真实数据') ## 在第一列，加入列（new_col2），并赋值为 “真实数据”
    # test_identity_analysis['new_col1'] = '真实数据' ## 新增一列数据在末尾，并赋值为 “真实数据”
    test_identity_analysis_part01 = test_identity_analysis[['id_01','id_02']]
    # print(test_identity_analysis_part01)


    # 列操作

    # 行操作
    # test_identity_analysis.loc[0] = 1
    test_identity_analysis = test_identity_analysis.append(test_identity.dtypes,ignore_index=True)
    test_identity_analysis['new_col'].loc[3]= '数据类型'

    test_identity_analysis = test_identity_analysis.append(test_identity.isnull().sum(),ignore_index=True)
    test_identity_analysis['new_col'].loc[4]= '空值数量'

    test_identity_analysis = test_identity_analysis.append(test_identity.min(),ignore_index=True)
    test_identity_analysis['new_col'].loc[5]= '最小值'

    test_identity_analysis = test_identity_analysis.append(test_identity.max(),ignore_index=True)
    test_identity_analysis['new_col'].loc[6]= '最大值'

    test_identity_analysis_part02 = test_identity_analysis.head(3)
    test_identity_analysis = test_identity_analysis.drop([0,1,2])
    test_identity_analysis = test_identity_analysis.append(test_identity_analysis_part02,ignore_index=True)

    # print(test_identity.describe())
    print(test_identity_analysis)



    # test_identity_analysis = test_identity_analysis.append(test_identity.describe(),ignore_index=True)




    print(test_identity_analysis)

    print(type(test_identity.dtypes))
    print(test_identity_analysis)

    # 数据概览
    print(test_identity.head(10))
    print(test_identity_analysis)
    print(test_identity_analysis['TransactionID'].loc[2])   # 显示TransactionID列，第二行数据
    print(test_identity_analysis[['TransactionID','id_01','id_02']])   # 显示TransactionID列，第二行数据
    print(type(test_identity_analysis[['TransactionID','id_01','id_02']].loc[2]))   # 显示TransactionID列，第二行数据
    print(test_identity_analysis.iloc[0:1,1:2]) # 显示第一行，第一列数据
    print(test_identity.tail())
    print(test_identity.sample())
    print(test_identity.shape)
    print(test_identity.dtypes)
    print(test_identity.axes)
    print(test_identity.min())
    print(test_identity.max())


    print(test_identity.info())
    print(test_identity.describe())

    print(test_identity.isnull().sum())

    #缺失值处理

    # print(test_identity.isnull)

    # df是表格名
    print(test_identity.loc[:, test_identity.isnull().any()])  # 输出存在空值的列
    print(test_identity.loc[:, test_identity.isnull().all()])  # 输出全为空值的列

    # df = pd.DataFrame(test_identity)
    # null_data = df[df.isnull().any(axis=1)]
    #
    # print(null_data)



    #空值处理
    #异常值处理

    # test_transaction = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\test_transaction.csv")
    # train_transaction = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\train_transaction.csv")
    # test_transaction.info()
    # train_transaction.info()





