import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
#warnings.filterwarnings('ignore')
#%matplotlib inline
from sklearn.metrics import roc_auc_score
## 数据降维处理的
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost
from sklearn.model_selection import train_test_split, cross_validate


train=pd.read_csv("./data/train.csv")
test=pd.read_csv("./data/test.csv")
sub=pd.read_csv("./data/submission.csv")

## 数据拼接
data=pd.concat([train,test])

## 时间转换
data['incident_date'] = pd.to_datetime(data['incident_date'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2022-06-30', '%Y-%m-%d')
data['time'] = data['incident_date'].apply(lambda x: startdate-x).dt.days
print(data)

## 编码
#Encoder
numerical_fea = list(data.select_dtypes(include=['object']).columns)
division_le = LabelEncoder()
for fea in numerical_fea:
    division_le.fit(data[fea].values)
    data[fea] = division_le.transform(data[fea].values)
print("数据预处理完成!")

## 数据集划分
testA=data[data['fraud'].isnull()].drop(['policy_id','incident_date','fraud'],axis=1)
trainA=data[data['fraud'].notnull()]
data_x=trainA.drop(['policy_id','incident_date','fraud'],axis=1)
data_y=train[['fraud']].copy()
col=['policy_state','insured_sex','insured_education_level','incident_type','collision_type','incident_severity','authorities_contacted','incident_state',
     'incident_city','police_report_available','auto_make','auto_model']
# for i in data_x.columns:
#     if i in col:
#         data_x[i] = data_x[i].astype('str')
# for i in testA.columns:
#     if i in col:
#         testA[i] = testA[i].astype('str')



model=CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="CPU",
            learning_rate=0.1,
            iterations=10000,
            random_seed=2020,
            od_type="Iter",
            depth=7,
            early_stopping_rounds=300)


answers = []
mean_score = 0
n_folds = 10
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)

X_train,X_test,y_train,y_test = train_test_split(data_x,data_y,train_size=0.7)


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
    score = cross_validate(val,X_test,y_test,cv=6,scoring='accuracy')
    K_result.loc[i,'accuracy'] = score['test_score'].mean()

K_result.index = pd.Series(['Tree','GBDT','XGBoost'])
print(K_result)


sub_df = pd.read_csv('./data/submission.csv')
sub_df['fraud'] = xgboost_clf.predict_proba(testA)[:,1]
sub_df.to_csv('./data/model_v20240430_1718.csv',index=False)
