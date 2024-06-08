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

    ## 数据清洗-Credit_Product空值填充
    X_test['Credit_Product'].fillna(value='No', inplace=True)  ## value代表用于填充的值，inplace代表是否在原数据集上进行修改
    X_train['Credit_Product'].fillna(value='No', inplace=True) ## value代表用于填充的值，inplace代表是否在原数据集上进行修改

