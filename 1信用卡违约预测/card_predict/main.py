
# 相关库的版本: pip install -r ./requeriment.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if __name__ == '__main__':
    import pandas as pd

    ## 0. 打印设置
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)  ## 显示全部结果，不带省略点
    pd.set_option('display.width',1000)
    pd.set_option('display.float_format', '{:.0f}'.format)

    ## 1. pandas读取文件
    X_test = pd.read_csv(r".\data\test.csv")
    X_train = pd.read_csv(r".\data\train.csv")





    ## 2. 数据预览

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
            overview.loc[col, 'nunique_nums'] = dataframe[col].nunique()

        print('----------------整体概况----------------')
        print(overview)


    overViewAnalysis1(X_test)
    overViewAnalysis1(X_train)

    # for analysis in [X_test, X_train]:
    #     print('----------------analysis----------------')
    #     print(analysis.shape)
    #     i = 50
    #     analysis_base = analysis
    #     analysis = analysis.head(i)
    #     analysis.insert(0, 'new_col', '真实数据')  ## 在第一列，加入列（new_col2），并赋值为 “真实数据”
    #     analysis = analysis._append(analysis_base.dtypes, ignore_index=True)
    #     analysis['new_col'].loc[int(i)] = '数据类型'
    #     analysis = analysis._append(analysis_base.isnull().sum(), ignore_index=True)
    #     analysis['new_col'].loc[int(i + 1)] = '空值数量'
    #     analysis = analysis._append(analysis_base.min(), ignore_index=True)
    #     analysis['new_col'].loc[int(i + 2)] = '最小值'
    #     analysis = analysis._append(analysis_base.max(), ignore_index=True)
    #     analysis['new_col'].loc[int(i + 3)] = '最大值'
    #     analysis = analysis._append(analysis_base.describe().loc['mean'], ignore_index=True)
    #     analysis['new_col'].loc[int(i + 4)] = '平均值'
    #     analysis = analysis._append(analysis_base.describe().loc['std'], ignore_index=True)
    #     analysis['new_col'].loc[int(i + 5)] = '方差'
    #
    #     analysis_part02 = analysis.head(i)
    #     # analysis = analysis.drop([0,1,2]) # 删除指定列
    #     analysis.drop(analysis.index[0:i], inplace=True)  # 删除切片行
    #     analysis = analysis.append(analysis_part02, ignore_index=True)
    #
    #     # print(test_identity.describe())
    #     print(analysis)


    ## 3. 数据清洗-Credit_Product空值填充
    X_test['Credit_Product'].fillna(value='No', inplace=True)  ## value代表用于填充的值，inplace代表是否在原数据集上进行修改
    X_train['Credit_Product'].fillna(value='No', inplace=True) ## value代表用于填充的值，inplace代表是否在原数据集上进行修改

    ## 4. 数据EDA分析
    import matplotlib.pyplot as plt
    import seaborn as sns
    #% matplotlib inline

    # 4.1 按年龄分布查看
    ages = [22, 30, 40, 50, 60, 70, 80, 90]
    df1 = X_train[X_train['Credit_Product']=='Yes'] ## 筛选 Credit_Product 为yes的
    binning = pd.cut(df1['Age'], ages, right=False)  ## 将连续量切分为离散变量 参考：https://blog.csdn.net/heianduck/article/details/124409593
    time = pd.value_counts(binning) ## 统计切分后，各个区间出现的次数

    # 4.2 可视化
    time = time.sort_index()  ## 根据数据标签排序
    fig = plt.figure(figsize=(6, 2), dpi=120)  ## 画布设置(dpi代表分辨率) https://blog.csdn.net/ximu__l/article/details/128588025
    sns.barplot(time.index, time, color='royalblue')  ## 条形图展示(x,y)   https://blog.csdn.net/Artoria_QZH/article/details/102768817
    # plt.show()
    import numpy as np
    x = np.arange(len(time)) ## np.arange(([start,] stop[, step,], dtype=None) 创建等差数列数组 https://www.python100.com/html/98340.html
    y = time.values   ## time:<class 'pandas.core.series.Series'> y:<class 'numpy.ndarray'>

    for x_loc, jobs in zip(x, y):  ## zip函数：组合x_loc和jobs为一个元组 https://blog.csdn.net/csdn15698845876/article/details/73411541
        plt.text(x_loc, jobs + 2, '{:.1f}%'.format(jobs / sum(time) * 100), ha='center', va='bottom', fontsize=8) ## plt.text(x,y,desc)数值显示，https://blog.csdn.net/zengbowengood/article/details/104324293

    plt.xticks(fontsize=8) ## 横坐标显示 例如：plt.xticks(x[::5], x_label[::5])
    plt.yticks([]) ## 横坐标显示 例如：plt.yticks(z[::5])
    plt.ylabel('') ##  纵轴标签
    plt.title('duration_yes', size=8)  ## 标题：https://www.jb51.net/article/279299.htm
    sns.despine(left=True) ## 移除坐标轴
    plt.show()


    # 1）分离数值变量与分类变量
    Nu_feature = list(X_train.select_dtypes(exclude=['object']).columns)  ## 选择数值类型不为OBJECT的列，参考：https://baijiahao.baidu.com/s?id=1769962569043539364&wfr=spider&for=pc
    Ca_feature = list(X_train.select_dtypes(include=['object']).columns)  ## 选择数值类型为OBJECT的列，参考：https://baijiahao.baidu.com/s?id=1769962569043539364&wfr=spider&for=pc

    ## import warnings

    ## warnings.filterwarnings("ignore")  ## 忽略全部警告

    ## plt.clf() ## 清空画布
    plt.figure(figsize=(15, 5))
    Nu_feature.remove('Target')

    # 2）根据数值型分布查看
    i = 1
    for col in Nu_feature:
        ax = plt.subplot(1, 3, i)   ## 创建小图1行3列第i个位置，https://blog.csdn.net/mouselet3/article/details/127389508
        ## 核密度估计图，主要用于查看连续数值的分布概率
        ax = sns.kdeplot(X_train[col], color='red')  ## 核密度估计图，参考：https://blog.csdn.net/fightingoyo/article/details/106873293；核密度估计：https://baike.baidu.com/item/%E6%A0%B8%E5%AF%86%E5%BA%A6%E4%BC%B0%E8%AE%A1/10349033
        ax = sns.kdeplot(X_test[col], color='cyan')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax = ax.legend(['train', 'test'])
        i += 1
    plt.show()

##  3）离散图，运行不出来，待优化
    # col1 = Ca_feature
    # plt.figure(figsize=(20, 10))
    # j = 1
    # for col in col1:
    #     ax = plt.subplot(6, 3, j)  ## 创建小图
    #     ax = plt.scatter(x=range(len(X_train)), y=X_train[col], color='red')  ## len(X_train):195725; range(len(X_train))=range(0, 195725); 散点图，参考：https://blog.csdn.net/gongdiwudu/article/details/129947219
    #     # plt.title(col) ## 设置标题：https://www.jb51.net/article/279299.htm
    #     j += 1
    #
    # k = 7
    # for col in col1:
    #     ax = plt.subplot(6, 3, k)
    #     ax = plt.scatter(x=range(len(X_test)), y=X_test[col], color='cyan')
    #     # plt.title(col)
    #     k += 1
    # plt.subplots_adjust(wspace=0.4, hspace=0.3)
    # plt.show()

    # 5. 离散数据Encoder
    from sklearn.preprocessing import LabelEncoder

    lb = LabelEncoder()    ## 使用labelebcode进行编码，参考：https://blog.csdn.net/m0_47256162/article/details/113788166
    cols = Ca_feature
    for m in cols:
        X_train[m] = lb.fit_transform(X_train[m])  ## 对非数值字段进行 labelcoder 转换，参考：https://blog.csdn.net/weixin_44027006/article/details/106160743
        X_test[m] = lb.fit_transform(X_test[m])

    print(X_train.head())
    print(X_test.head())

    correlation_matrix = X_train.corr() ## corr()函数查找行之间的相关性
    print(correlation_matrix)
    plt.figure(figsize=(12, 10))
    # 热力图
    sns.heatmap(correlation_matrix, vmax=0.9, linewidths=0.05, cmap="RdGy") ## 绘制热力图，是识别预测变量与目标变量相关性的方法; 参考：https://blog.csdn.net/weixin_46649052/article/details/115231716
    plt.show()

    ## 6. 建立模型
    ## 6.1 切割训练集和测试集
    X=X_train.drop(columns=['ID','Target'])
    print(X)
    Y=X_train['Target']
    X_test=X_test.drop(columns='ID')

    # 数据切分
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    from lightgbm.sklearn import LGBMClassifier

    ## 6.2 创建模型
    gbm = LGBMClassifier(n_estimators=600, learning_rate=0.01, boosting_type='gbdt',  ## 模型训练超参数 调优参考：https://blog.51cto.com/u_16213313/7201851
                         objective='binary',   ## LGBMClassifier详解： https://blog.csdn.net/yeshang_lady/article/details/118638269
                         max_depth=-1,
                         random_state=2022,
                         metric='auc')


    ## 7. 模型训练
    # 7.1 交叉验证
    result1 = []
    mean_score1 = 0
    n_folds = 5

    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, auc, roc_auc_score


    kf = KFold(n_splits=n_folds, shuffle=True, random_state=2022)  ## 交叉验证，参考：https://blog.csdn.net/m0_67173953/article/details/132414215
    print(kf)
    print(type(kf))
    print(type(kf.split(X)))
    print(kf.split(X))
    for train_index, test_index in kf.split(X): ## KFold.split()方法接受一个数据集（通常是一个数组或DataFrame），并返回一个迭代器，该迭代器产生K个元组。 参考：https://blog.csdn.net/AdamCY888/article/details/134742894
        print(train_index)
        print(test_index)
        x_train = X.iloc[train_index]  ## 拿取K-1份数据
        y_train = Y.iloc[train_index]
        print(x_train)
        print(y_train)
        x_test = X.iloc[test_index]
        y_test = Y.iloc[test_index]
        print(x_test)
        print(y_test)
        gbm.fit(x_train, y_train) ## 开始训练
        y_pred1 = gbm.predict_proba((x_test), num_iteration=gbm.best_iteration_)[:, 1]  ## 开始预测
        print('验证集AUC:{}'.format(roc_auc_score(y_test, y_pred1)))          ## roc & auc 的理解，参考：https://blog.csdn.net/sereasuesue/article/details/108940876



        mean_score1 += roc_auc_score(y_test, y_pred1) / n_folds
        y_pred_final1 = gbm.predict_proba((X_test), num_iteration=gbm.best_iteration_)[:, 1]
        y_pred_test1 = y_pred_final1
        result1.append(y_pred_test1)

    # 7.2 模型评估
    print('mean 验证集auc:{}'.format(mean_score1))
    cat_pre1 = sum(result1) / n_folds

    ret1 = pd.DataFrame(cat_pre1, columns=['Target'])
    ret1['Target'] = np.where(ret1['Target'] > 0.5, '1', '0').astype('str')


    result = pd.DataFrame()
    test = pd.read_csv(r"D:\wk\myProject\dataMining_fraudPredict\信用卡违约预测\card_predict\data\test.csv")
    result['ID'] = test['ID']
    result['Target'] = ret1['Target']

    result.to_csv(r"D:\wk\myProject\dataMining_fraudPredict\信用卡违约预测\card_predict\data\result_v20240609.csv", index=False)






















