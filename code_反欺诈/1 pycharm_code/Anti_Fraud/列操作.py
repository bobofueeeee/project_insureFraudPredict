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
    print(test_identity_analysis_part01)
