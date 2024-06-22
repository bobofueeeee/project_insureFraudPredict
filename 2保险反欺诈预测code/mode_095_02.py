# 导入所需要的库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import warnings



warnings.filterwarnings('ignore')
import datetime as dt

import xgboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_curve # 绘制ROC曲线
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
# %matplotlib inline



## 一 EDA

# 导入数据
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
financial_data = train_data
print(financial_data)# 查看缺失值

numercial_feature = list(financial_data.select_dtypes(exclude=['object']).columns) #数值型变量
object_feature = list(financial_data.select_dtypes(include=['object']).columns)
print(numercial_feature,object_feature)

## 1.数据结构探索
### 区分离散型变量和连续型变量

numercial_feature = list(financial_data.select_dtypes(exclude=['object']).columns) #数值型变量
object_feature = list(financial_data.select_dtypes(include=['object']).columns)
print(numercial_feature,object_feature)

# 连续型变量
serial_feature = []
# 离散型变量
discrete_feature = []
# 单值变量
unique_feature = []

for feature in numercial_feature:
    temp = financial_data[feature].nunique() #返回数据去重后的个数
    if temp == 1:
        unique_feature.append(feature)
    elif temp >1 and temp <= 10:
        discrete_feature.append(feature)
    else:
        serial_feature.append(feature)

## 2 连续型变量
serial_df = pd.melt(financial_data,value_vars=serial_feature) #将连续型变量融合在一个dataframe中
f = sns.FacetGrid(serial_df,col='variable',col_wrap=3, sharex=False, sharey=False) # 生成画布，最多三列，不共享x、y轴
f.map(sns.distplot,"value")

# 单独查看每年保费分布
plt.figure(figsize=(10,5))
sns.kdeplot(financial_data.policy_annual_premium[financial_data['fraud'] == 1],shade=True)# 违约者
sns.kdeplot(financial_data.policy_annual_premium[financial_data['fraud'] == 0],shade=True)# 没有违约
plt.xlabel('policy_annual_premium')
plt.ylabel('Density')

# 查看资本利得损失变化
fig,axes = plt.subplots(1,2,figsize=(15,6))
# sns.kdeplot(financial_data['capital-gains'],hue=financial_data['fraud'],shade=True,ax=axes[0])
# sns.kdeplot(financial_data['capital-loss'],hue=financial_data['fraud'],shade=True,ax=axes[1])

plt.figure(figsize=(10,5))
sns.kdeplot(financial_data.umbrella_limit[financial_data['fraud'] == 1],shade=True)# 违约者
sns.kdeplot(financial_data.umbrella_limit[financial_data['fraud'] == 0],shade=True)# 没有违约
plt.xlabel('umbrella_limit')
plt.ylabel('Density')

## 3.离散型变量
for val in discrete_feature:
    temp = financial_data[val].nunique()
    print(val,'类型数',temp)

# 绘制频数图
discrete_df = financial_data[discrete_feature]
fig,axes = plt.subplots(1,5,figsize=(20,5))
sns.set_style('whitegrid')
for i,val in enumerate(discrete_feature):
    sns.countplot(data=discrete_df,x=val,hue='fraud',ax=axes[i])

## 4.分类型变量
object_feature
financial_data[object_feature]
# 变量有日期型、缺失值
financial_data[object_feature].isna().any(axis=0)
# collision_type、property_damage、police_report_available有缺失值

category_df = pd.concat([financial_data[object_feature], financial_data[discrete_feature[-1]]], axis=1)
n = len(object_feature) // 6
# fig,axes = plt.subplots(1,n,figsize=(20,5))
sns.set_style('whitegrid')
# for i,val in enumerate(object_feature[0:n]):
#    sns.countplot(data=category_df,x=val,hue='fraud',ax=axes[i],order=category_df[val].value_counts().index)
#    plt.xticks(rotation=60)

# fig,axes = plt.subplots(1,n,figsize=(20,5))
# for i,val in enumerate(object_feature[n:n+n]):
#    sns.countplot(data=category_df,x=val,hue='fraud',ax=axes[i],order=category_df[val].value_counts().index)
#    plt.xticks(rotation=60)

# fig,axes = plt.subplots(1,n,figsize=(20,5))
# for i,val in enumerate(object_feature[2*n:3*n-1]):
#    sns.countplot(data=category_df,x=val,hue='fraud',ax=axes[i],order=category_df[val].value_counts().index)
#    plt.xticks(rotation=60)

## 6.缺失值的处理
X_missing = financial_data.drop(columns='fraud')
missing = X_missing.isna().mean()
missing_df = pd.DataFrame({'missing_key':missing.keys(),'missing_value':np.round(missing.values,4)})
plt.figure(figsize=(20,10))
sns.barplot(data=missing_df,x='missing_key',y='missing_value')
plt.xticks(rotation=90)
# 缺失值都不超过50%，所以我们进行填充缺失值

## 7.变量相关性
cor = financial_data[numercial_feature].corr()
# sns.set_theme(style="white")
plt.figure(figsize=(16,8))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(cor,cmap=cmap,annot = True,linewidth = 0.2,
            cbar_kws={"shrink": .5},linecolor = "white",fmt =".1g")

## 三、特征工程
### 1.缺失值的填充
### 对于分类型变量使用众数进行填充
### 对于连续型变量使用均值或中位数进行填充，区别在于分布是否偏态
# 先去除标签列
label = 'fraud'
numercial_feature.remove(label)

missing_feature = list(missing_df[missing_df['missing_value'] != 0].missing_key) # 有缺失值的特征
print(financial_data[missing_feature]) # 可以看出来都是分类型变量
for val in missing_feature:
    train_data[val] = train_data[val].fillna(train_data[val].mode()[0])
    test_data[val] = test_data[val].fillna(test_data[val].mode()[0])

    ### 2.异常值的处理
    ### 异常值通过3$\sigma$或者箱线图来确定
f_box = sns.FacetGrid(serial_df, col='variable', col_wrap=3, sharex=False, sharey=False)  # 生成画布，最多三列，不共享x、y轴
f_box.map(sns.boxplot, "value")  # 发现存在变量有异常 由于是特殊的风险预测，所以保留异常值

### 3.时间数据的处理
### 主要对日期进行拆分
datetime_list = ['policy_bind_date','incident_date']
financial_data[['policy_bind_date','incident_date']]
# 我们基于出险日期和保险绑定日期做一个差值，并取出出险日期作为单独的字段

import datetime

for val in datetime_list:
    train_data[val] = pd.to_datetime(train_data[val],format='%Y-%m-%d')
    test_data[val] = pd.to_datetime(test_data[val], format='%Y-%m-%d')
# 转化时间格式
# 创建新特征，两个日期之差
train_data['detla_time'] = (train_data['incident_date'] - train_data['policy_bind_date']).dt.days
test_data['detla_time'] = (test_data['incident_date'] - test_data['policy_bind_date']).dt.days

fig,axes = plt.subplots(1,2,figsize=(16,8))
sns.kdeplot(train_data['detla_time'],shade=True,ax=axes[0])
sns.kdeplot(test_data['detla_time'],shade=True,ax=axes[1])
# 分布良好

train_data['picked_month'] = train_data['incident_date'].dt.month
test_data['picked_month'] = test_data['incident_date'].dt.month
train_data['picked_month'] = train_data['picked_month'].apply(lambda x: str(x) + '月')
test_data['picked_month'] = test_data['picked_month'].apply(lambda x: str(x) + '月')
# 将数字处理成字符，因为我们不在意大小而在于出现的频次

train_data.drop(columns=['policy_bind_date','incident_date'],inplace=True)
test_data.drop(columns=['policy_bind_date','incident_date'],inplace=True)# 删除两个日期型feature

object_feature.remove('policy_bind_date')
object_feature.remove('incident_date')
object_feature.append('picked_month')
numercial_feature.append('detla_time')

### 4.特征选择
train_data = train_data.drop(columns='policy_id')
test_data = test_data.drop(columns='policy_id')# 删除id字段
train_data.corr()['fraud'].sort_values # 查看变量与fraud的相关系数
# 决定全部保留

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

getHighRelatedFeatureDf(train_data.corr(),0.6)
# age和moth有明显相关性，删除age(与fraud相关性不高)
# 删除injury_claim

getHighRelatedFeatureDf(train_data.corr(),0.6)
# age和moth有明显相关性，删除age(与fraud相关性不高)
# 删除injury_claim

numercial_feature.remove('age')
numercial_feature.remove('policy_id')
numercial_feature.remove('injury_claim')

### 5.特征编码
y_label = train_data['fraud']
train_data = train_data.drop(columns='fraud')
print(train_data)



print(train_data[object_feature].nunique()) # 对十个以上的变量进行meanencoder)
train_data[object_feature].nunique() # 对十个以上的变量进行meanencoder
meancoder_list = ['insured_occupation','insured_hobbies','auto_make','auto_model','insured_education_level'
                  ,'incident_state','incident_city']
# 类别变量中去除要meanencoder编码的
for val in meancoder_list:
    object_feature.remove(val)


from my_module import MeanEncoder
meanencoder = MeanEncoder(categorical_features=meancoder_list,target_type='classification')
mean_X_train = meanencoder.fit_transform(train_data,y_label)
mean_X_test = meanencoder.transform(test_data)

mean_X_test.shape,mean_X_train.shape

# 去除meanencoder编码的变量
mean_X_train = mean_X_train.drop(columns=meancoder_list)
mean_X_test = mean_X_test.drop(columns=meancoder_list)

mean_X_test.shape,mean_X_train.shape

### 普通编码

# 对其余变量编码
from sklearn.preprocessing import LabelEncoder
Label = LabelEncoder()
for val in object_feature:
    Label.fit(mean_X_train[val])
    mean_X_train[val] = Label.transform(mean_X_train[val])
    mean_X_test[val] = Label.transform(mean_X_test[val])

## 四、构建模型
# 划分数据集为测试集和训练集
X_train,X_test,y_train,y_test = train_test_split(mean_X_train,y_label,train_size=0.7)
#simple_X_train,simple_X_test,simple_y_train,simple_y_test = train_test_split(simple_X,simple_y,train_size=0.7)

# 集合算法树模型
GBDT_param = {
    'loss': 'log_loss',
    'learning_rate': 0.1,
    'n_estimators': 30,
    'max_depth': 3,
    'min_samples_split': 300
}
GBDT_clf = GradientBoostingClassifier()  # GBDT模型

tree_param = {
    'criterion': 'gini',
    'max_depth': 30,
    'min_impurity_decrease': 0.1,
    'min_samples_leaf': 2

}
Tree_clf = DecisionTreeClassifier(**tree_param)  # 决策树模型

xgboost_param = {
    'learning_rate': 0.01,
    'reg_alpha': 0.,
    'max_depth': 3,
    'gamma': 0,
    'min_child_weight': 1

}
xgboost_clf = xgboost.XGBClassifier(**xgboost_param)  # xgboost模型

xgboost_clf.fit(X_train, y_train)
GBDT_clf.fit(X_train, y_train)
Tree_clf.fit(X_train, y_train)

# K折交叉检验
K_model_list = [Tree_clf,GBDT_clf,xgboost_clf]
K_result = pd.DataFrame()
for i,val in enumerate(K_model_list):
    print(i)
    print(val)
    score = cross_validate(val,mean_X_train,y_label,cv=6,scoring='accuracy')
    K_result.loc[i,'accuracy'] = score['test_score'].mean()
K_result.index = pd.Series(['Tree','GBDT','XGBoost'])
print(K_result)

# 特征重要性排序
from xgboost import plot_importance
fig,ax = plt.subplots(figsize=(20,8))
plot_importance(xgboost_clf,ax=ax)

## 提交结果
sub_df = pd.read_csv('./data/submission.csv')
sub_df['fraud'] = xgboost_clf.predict_proba(mean_X_test)[:,1]
sub_df.to_csv('./data/model_095_02.csv',index=False)