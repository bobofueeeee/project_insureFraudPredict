import pandas as pd

if __name__ == '__main__':

    # 创建一个DataFrame，假设它已经有一些行数据
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    print(df)

    df01 = df.head(2)
    print(df01)

    # 添加新数据到末尾
    new_row = pd.Series({'col1': 5, 'col2': 6})
    df01 = df01.append(new_row, ignore_index=True)
    print(df01)

    data = {'name': ['Tom', 'Jerry'], 'age': [20, 21]}

    df = pd.DataFrame(data)

    new_row = pd.Series({'id_01': 1.0, 'id_02': 6.0})
    new_data = pd.Series({'name': 'Mike', 'age': 22})

    df = df.append(new_data, ignore_index=True)

    print(df)

    test_identity = pd.read_csv(r"C:\Users\a\Desktop\wk\0 个人数据挖掘项目计划\0 数据集\raw_data\test_identity.csv")
    test_identity_analysis = test_identity.head(3)

    test_identity_analysis_part01 = test_identity_analysis[['id_01','id_02']]
    print(test_identity_analysis_part01)

    new_row = pd.Series({'id_01': 1.0, 'id_02': 6.0})
    test_identity_analysis_part01.append(new_row,ignore_index=True)
    print(test_identity_analysis_part01)
