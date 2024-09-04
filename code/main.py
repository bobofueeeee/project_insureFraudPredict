import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.model_selection import KFold
## 0. 打印设置
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)  ## 显示全部结果，不带省略点
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)
import pandas as pd
from matplotlib import  pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
import xgboost
import datetime
# %matplotlib inline

# 1. 概况分析
# 1.1 读取数据
x_train = pd.read_csv(r'./data/train.csv')
x_test = pd.read_csv(r'./data/test.csv')

# 1.2 概况预览
def overViewAnalysis1(dataframe):
    print('----------------整体概况----------------')
    overview = pd.DataFrame()
    overview['type'] = dataframe.dtypes
    overview['row_nums'] = dataframe.shape[0]
    overview['null_nums'] = dataframe.isnull().sum()
    # overview['min_num'] = dataframe.min()
    # overview['max_num'] = dataframe.max()
    overview['mean_num'] = dataframe.describe().loc['mean']
#     overview['std_num'] = dataframe.describe().loc['std']

    for col in dataframe.columns:
        overview.loc[col,'nunique_nums'] = dataframe[col].nunique()

    print(overview)
    print('----------------整体概况----------------')

overViewAnalysis1(x_train)
overViewAnalysis1(x_test)

## 2. 数据EDA探索
## 2.1 数据清洗
# 0） 空值处理，不需要
# 1) 日期转换,加入后，得分下降
# policy_bind_date, incident_date
for data in x_train, x_test:
    data['incident_date'] = pd.to_datetime(data['incident_date'], format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2022-06-30', '%Y-%m-%d')
    data['time'] = data['incident_date'].apply(lambda x: startdate - x).dt.days

overViewAnalysis1(x_train)
overViewAnalysis1(x_test)

# 2） 数据编码
Ca_feature = list(x_train.select_dtypes(include=['object']).columns)
lb = LabelEncoder()
for col in Ca_feature:
    x_train[col] = lb.fit_transform(x_train[col])
    x_test[col] = lb.fit_transform(x_test[col])
overViewAnalysis1(x_train)

## 2.2 查看数据分布，查看数据分布是为了干什么，不懂，先跳过
## 2.3 查看相关性,生成热力图
cor = x_train.corr()
plt.figure(figsize=(40,20))
cmap = sns.diverging_palette(120, 10, as_cmap=True)
sns.heatmap(cor,cmap=cmap,annot = True,linewidth = 0.2,cbar_kws={"shrink": .5},linecolor = "white",fmt =".1g")
plt.show()

# 显示相关性高于0.6的变量
def getHighRelatedFeatureDf(corr_matrix, corr_threshold):
    highRelatedFeatureDf = pd.DataFrame(corr_matrix[corr_matrix > corr_threshold].stack().reset_index())
    highRelatedFeatureDf.rename({'level_0':'feature1','level_1':'feature2',0:'corr'},axis=1,inplace=True) # 更改列名
    highRelatedFeatureDf = highRelatedFeatureDf[highRelatedFeatureDf.feature1 != highRelatedFeatureDf.feature2] # 去除自己和自己
    highRelatedFeatureDf['feature_pair_key'] = highRelatedFeatureDf.loc[:,['feature1', 'feature2']].apply(lambda r:'#'.join(np.sort(r.values)), axis=1)
    # 将feature1和feature2名称连接在一起去重
    highRelatedFeatureDf.drop_duplicates(subset=['feature_pair_key'],inplace=True)
    highRelatedFeatureDf.drop(columns='feature_pair_key',inplace=True)
    return highRelatedFeatureDf

print(getHighRelatedFeatureDf(cor,0.6))

## 去除相关性高的特征
for col in ['incident_date','age']: # ,
    del x_train[col]
    del x_test[col]

## 3. 样本划分
y_train = x_train['fraud']
x_train = x_train.drop(columns=['policy_id', 'fraud'])

# x_train_01,x_train_02,y_train_01,y_train_02 = train_test_split(x_train,y_train,train_size=0.7)

# x_train_01,x_train_02,y_train_01,y_train_02 = train_test_split(x_train,y_train,test_size=0.3,random_state=42)

# x_train_01, x_train_02 = train_test_split(x_train, test_size=0.3,random_state=42)  # 25% of remaining data as validation set
# y_train_01, y_train_02 = train_test_split(y_train, test_size=0.3, random_state=42)  # Split labels accordingly

## 4. 模型训练
# 4.1 GBDT模型
GBDT_param = {
    'loss': 'log_loss',
    'learning_rate': 0.1,
    'n_estimators': 30,
    'max_depth': 3,
    'min_samples_split': 300
}

GBDT_clf = GradientBoostingClassifier()

# 4.2 决策树模型
tree_param = {
    'criterion': 'gini',
    'max_depth': 30,
    'min_impurity_decrease': 0.1,
    'min_samples_leaf': 2

}
Tree_clf = DecisionTreeClassifier(**tree_param)  #

# 4.3 xgboost模型
xgboost_param = {
    'learning_rate': 0.01,
    'reg_alpha': 0.,
    'max_depth': 3,
    'gamma': 0,
    'min_child_weight': 1
}

xgboost_clf = xgboost.XGBClassifier(**xgboost_param)

# xgboost_clf.fit(x_train_01, y_train_01)
# GBDT_clf.fit(x_train_01, y_train_01)
# Tree_clf.fit(x_train_01, y_train_01)

answers = []
mean_score = 0
n_folds = 10
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)
for train, test in sk.split(x_train, y_train):
    x_train_01 = x_train.iloc[train]
    y_train_01 = y_train.iloc[train]
    x_train_02 = x_train.iloc[test]
    y_train_02 = y_train.iloc[test]

    xgboost_clf.fit(x_train_01, y_train_01)
    #     clf = model.fit(x_train,y_train, eval_set=(x_test,y_test),verbose=500,cat_features=col)
    y_train_02_pred = xgboost_clf.predict_proba(x_train_02)[:, -1]
    print('cat验证的auc:{}'.format(roc_auc_score(y_train_02, y_train_02_pred)))
    mean_score += roc_auc_score(y_train_02, y_train_02_pred) / n_folds  ## 将n_folds次的auc值求了一个平均值
    print('--------mean_score---------')
    print(mean_score)

    y_test_pred = xgboost_clf.predict_proba(x_test.drop(columns=['policy_id']))[:, -1]
    print('--------y_pred_valid---------')
    print(y_test_pred)
    answers.append(y_test_pred)
    print('--------answers---------')
    print(answers)

print('10折平均Auc:{}'.format(mean_score))

# 5. 模型评估
# 5.1 K折交叉检验
K_model_list = [xgboost_clf] # Tree_clf,GBDT_clf
K_result = pd.DataFrame()
for i,val in enumerate(K_model_list):
    score = cross_validate(val,x_train,y_train,cv=6,scoring='accuracy')
    K_result.loc[i,'accuracy'] = score['test_score'].mean()
K_result.index = pd.Series(['XGBoost']) # 'Tree','GBDT',
print(K_result)

## 5.2 指标重要性排序
from xgboost import plot_importance
fig,ax = plt.subplots(figsize=(20,8))
plot_importance(xgboost_clf,ax=ax)

## 6. 结果输出
## 6.1 测试集测试

y_test_pred=sum(answers)/n_folds
model_name = 'model_v20240430_1810'
result = pd.read_csv('./data/submission.csv')
result['fraud'] = y_test_pred
result.to_csv(f'./data/{model_name}.csv', index=False)