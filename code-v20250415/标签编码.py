import pandas as pd
from sklearn.preprocessing import LabelEncoder

X_train = pd.read_csv(r"D:\wk\myProject\project_insureFraudPredict\data\train.csv")
object_feature = list(X_train.select_dtypes(include=['object']).columns)
lb = LabelEncoder()
for col in object_feature:
    X_train[col] = lb.fit_transform(X_train[col])
print(X_train)