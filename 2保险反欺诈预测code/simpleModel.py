from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成一个二元分类问题的数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用逻辑回归作为分类器
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)

# 预测概率
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# 计算AUC值
auc = roc_auc_score(y_test, y_pred_prob)
print(f'AUC: {auc:.2f}')