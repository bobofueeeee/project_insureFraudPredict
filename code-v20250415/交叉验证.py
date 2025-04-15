from sklearn.model_selection import StratifiedKFold
import numpy as np

# 假设我们有一个特征矩阵 X 和一个目标向量 y
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([0, 0, 1, 1, 0, 1])  # 这是一个二分类问题的目标变量

# 定义交叉验证的参数
n_folds = 3  # 折数

# 初始化 StratifiedKFold 对象
sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)

# 使用 StratifiedKFold 进行交叉验证
for train_index, test_index in sk.split(X, y):
    print(test_index)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Train indices:", train_index, "Test indices:", test_index)
    print("X_train:", X_train, "X_test:", X_test)
    print("y_train:", y_train, "y_test:", y_test)
    print()  # 空行用于分隔不同的折


# 参数解释
# n_splits=n_folds：指定交叉验证的折数。在这个例子中，我们设置了 n_folds=3，意味着数据将被分成3个折，每次使用其中2个折作为训练集，1个折作为测试集，进行3次交叉验证。
# shuffle=True：在划分折之前是否对数据进行洗牌。洗牌有助于打破数据中的任何潜在顺序依赖，从而提高交叉验证的可靠性。
# random_state=2019：设置随机种子，以确保每次运行代码时都能得到相同的数据划分。这对于结果的可重复性非常重要。


# 使用场景
# 分类问题：StratifiedKFold 特别适用于分类问题，尤其是当类别不平衡时。
# 模型评估：通过交叉验证来评估模型的性能，确保模型在不同数据子集上的表现一致。
# 超参数调优：在交叉验证的循环中，可以使用网格搜索或随机搜索等方法来调优模型的超参数。


# 注意事项
# 如果你的数据集非常大，交叉验证可能会非常耗时。在这种情况下，可以考虑使用较小的折数或减少数据集的规模。
# 对于回归问题，通常使用 KFold 而不是 StratifiedKFold，因为回归问题中没有类别标签需要保持比例一致。