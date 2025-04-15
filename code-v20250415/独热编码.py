import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 示例数据：单个特征的多行样本
data = np.array(['red', 'green', 'blue', 'green', 'red'])

# 重塑数据为二维数组，其中每个样本有一个特征
data_reshaped = data.reshape(-1, 1)

X_train = pd.read_csv(r"D:\wk\myProject\project_insureFraudPredict\data\train.csv")

object_feature = list(X_train.select_dtypes(include=['object']).columns)

# 初始化OneHotEncoder
# onehot = OneHotEncoder(sparse_output=False, drop='first')  # sparse_output=False返回数组而不是稀疏矩阵，drop='first'可选，用于避免多重共线性

for col in object_feature:
    # 正确的单独处理示例（避免重复使用同一encoder实例fit）：
    onehot = OneHotEncoder(sparse_output=False, drop='first')
    if col not in ['incident_date','policy_bind_date']:
        # 注意：OneHotEncoder需要二维数组作为输入，因此我们需要对单个列进行reshape
        X_train[col] = onehot.fit_transform(X_train[[col]])
print(X_train)



