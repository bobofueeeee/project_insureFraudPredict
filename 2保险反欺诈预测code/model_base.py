import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

## 0. 打印设置
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)  ## 显示全部结果，不带省略点
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.0f}'.format)


def overViewAnalysis(list):
    for analysis in [list]:
        print('----------------行数和列数----------------')
        print(analysis.shape)
        i = 5
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
        print('----------------整体概况----------------')
        print(analysis)

def overViewAnalysis1(dataframe):
    print('----------------整体概况----------------')
    overview = pd.DataFrame()
    overview['type'] = dataframe.dtypes
    overview['row_nums'] = dataframe.shape[0]
    overview['null_nums'] = dataframe.isnull().sum()
    overview['min_num'] = dataframe.min()
    overview['max_num'] = dataframe.max()
    # overview['mean_num'] = dataframe.describe().loc['mean']
    overview['std_num'] = dataframe.describe().loc['std']

    for col in dataframe.columns:
        overview.loc[col,'nunique_nums'] = dataframe[col].nunique()

    print('----------------整体概况----------------')
    print(overview)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## 1. 读取数据
    test_Base = pd.read_csv(r"data/test.csv")
    train_Base = pd.read_csv(r"data/train.csv")

    overViewAnalysis1(test_Base)

    ## 2. 特征工程
    ## 2.1 特征编码
    Ca_feature = list(train_Base.select_dtypes(include=['object']).columns)
    lb = LabelEncoder()
    for m in Ca_feature:
        train_Base[m] = lb.fit_transform(train_Base[m])
        test_Base[m] = lb.fit_transform(test_Base[m])

    ## 2.2 数据集划分，将train划分为训练集和验证集

    train_X = train_Base.drop(columns=['policy_id', 'fraud'])
    train_Feature = train_Base['fraud']

    print(train_Feature)

    X_train, X_yanzheng = train_test_split(train_X, test_size=0.2,random_state=42)  # 25% of remaining data as validation set
    y_train, y_yanzheng = train_test_split(train_Feature, test_size=0.2, random_state=42)  # Split labels accordingly


    ## 3. 模型训练
    ## 3.1 建立模型
    gbm = LGBMClassifier(n_estimators=600, learning_rate=0.01, boosting_type='gbdt',
                         ## 模型训练超参数 调优参考：https://blog.51cto.com/u_16213313/7201851
                         objective='binary',
                         ## LGBMClassifier详解： https://blog.csdn.net/yeshang_lady/article/details/118638269
                         max_depth=-1,
                         random_state=2022,
                         metric='auc')

    ## 3.2 模型训练
    gbm.fit(X_train, y_train)

    ## 3.3 模型预测
    X_yanzheng_pred = gbm.predict_proba(X_yanzheng)
    print(X_yanzheng_pred)
    print('验证集AUC:{}'.format(roc_auc_score(y_yanzheng, X_yanzheng_pred[:, 1])))  ## ,multi_class='ovr'
    X_test = test_Base.drop(columns=['policy_id'])
    X_test_pred = gbm.predict_proba(X_test)

    # 3.4 预测结果转换与输出
    X_test_pred[:, 1][X_test_pred[:, 1] > 0.5] = '1'
    X_test_pred[:, 1][X_test_pred[:, 1] <= 0.5] = '0'
    print(X_test_pred[:, 1])
    submission_Base = pd.read_csv(r"data/submission.csv")
    submission_Base['fraud'] = X_test_pred[:, 1]
    print(submission_Base)

    # submission_Base.to_csv(r"data/submission_20240426.csv", index=False)
    ## score:0.8794