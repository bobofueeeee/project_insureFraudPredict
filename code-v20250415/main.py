import pandas as pd

if __name__ == '__main__':
    ## 打印设置
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.0f}'.format)

    ## pandas读取文件
    X_train = pd.read_csv(r"D:\wk\myProject\project_insureFraudPredict\data\train.csv")
    column_name = pd.read_excel(r"D:\wk\myProject\project_insureFraudPredict\data\column_name.xlsx")
    ## 数据预览
    for analysis in [X_train]:
        print('----------------analysis----------------')
        print(analysis.shape)
        i = 50
        analysis_base = analysis
        analysis = analysis.head(i)

        # 在第一列，加入列（new_col），并赋值为 “真实数据”
        analysis.insert(0, 'new_col', '真实数据')


        # 定义一个函数来创建单行 DataFrame，并确保列名对齐
        def create_row(index_name, col_value, original_df):
            row = pd.DataFrame([col_value], index=[index_name], columns=original_df.columns)
            row['new_col'] = index_name  # 确保 new_col 列存在
            return row


        # 使用 pd.concat 替代 append
        # 添加数据类型行
        analysis = pd.concat([analysis, create_row('数据类型', analysis_base.dtypes, analysis)], ignore_index=True)

        # 添加空值数量行
        analysis = pd.concat([analysis, create_row('空值数量', analysis_base.isnull().sum(), analysis)], ignore_index=True)

        # 添加最小值行
        min_value_seres = create_row('最小值', analysis_base.min(), analysis)
        analysis = pd.concat([analysis, create_row('最小值', analysis_base.min(), analysis)], ignore_index=True)

        # 添加最大值行
        analysis = pd.concat([analysis, create_row('最大值', analysis_base.max(), analysis)], ignore_index=True)

        # 添加平均值行
        mean_row = analysis_base.describe().loc['mean']
        analysis = pd.concat([analysis, create_row('平均值', mean_row, analysis)], ignore_index=True)

        # 添加方差行
        std_row = analysis_base.describe().loc['std']
        analysis = pd.concat([analysis, create_row('方差', std_row, analysis)], ignore_index=True)

        # 分离前 i 行
        analysis_part02 = analysis.head(i)

        # 删除前 i 行（实际上是 i+6 行，因为追加了 6 行统计信息）
        analysis = analysis.iloc[i + 6:].reset_index(drop=True)

        # 将前 i 行重新添加到最后
        analysis = pd.concat([analysis, analysis_part02], ignore_index=True)

        # 打印结果
        print(analysis)