if __name__ == '__main__':
    print("---------开始数据分析---------")
    import pandas as pd


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width',1000)
    pd.set_option('display.float_format', '{:.0f}'.format)

    test_identity = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\test_identity.csv")
    train_identity = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\train_identity.csv.csv")


    for analysis in [test_identity,train_identity]:

        i = 5
        analysis = analysis.head(i)
        test_identity_analysis.insert(0,'new_col','真实数据') ## 在第一列，加入列（new_col2），并赋值为 “真实数据”
        test_identity_analysis = test_identity_analysis.append(test_identity.dtypes,ignore_index=True)
        test_identity_analysis['new_col'].loc[int(i)]= '数据类型'
        test_identity_analysis = test_identity_analysis.append(test_identity.isnull().sum(),ignore_index=True)
        test_identity_analysis['new_col'].loc[int(i+1)]= '空值数量'
        test_identity_analysis = test_identity_analysis.append(test_identity.min(),ignore_index=True)
        test_identity_analysis['new_col'].loc[int(i+2)]= '最小值'
        test_identity_analysis = test_identity_analysis.append(test_identity.max(),ignore_index=True)
        test_identity_analysis['new_col'].loc[int(i+3)]= '最大值'
        test_identity_analysis = test_identity_analysis.append(test_identity.describe().loc['mean'],ignore_index=True)
        test_identity_analysis['new_col'].loc[int(i+4)]= '平均值'
        test_identity_analysis = test_identity_analysis.append(test_identity.describe().loc['std'],ignore_index=True)
        test_identity_analysis['new_col'].loc[int(i+5)]= '方差'

        test_identity_analysis_part02 = test_identity_analysis.head(i)
        # test_identity_analysis = test_identity_analysis.drop([0,1,2]) # 删除指定列
        test_identity_analysis.drop(test_identity_analysis.index[0:i], inplace=True)   # 删除切片行
        test_identity_analysis = test_identity_analysis.append(test_identity_analysis_part02,ignore_index=True)

        # print(test_identity.describe())
        print(test_identity_analysis)
        print(test_identity.shape)

    print("---------开始数据清洗---------")
