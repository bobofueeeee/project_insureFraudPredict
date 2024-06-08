
if __name__ == '__main__':
    import pandas as pd

    ## 打印设置
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)  ## 显示全部结果，不带省略点
    pd.set_option('display.width',1000)
    pd.set_option('display.float_format', '{:.0f}'.format)

    ## pandas读取文件
    X_test = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\2 信用卡违约预测\0 数据集\test.csv")
    X_train = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\2 信用卡违约预测\0 数据集\train.csv")

    ## 数据预览
    for analysis in [X_test, X_train]:
        print('----------------analysis----------------')
        print(analysis.shape)
        i = 50
        analysis_base = analysis
        analysis = analysis.head(i)
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




