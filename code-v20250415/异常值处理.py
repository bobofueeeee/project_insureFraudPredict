import pandas as pd
import numpy as np
from scipy import stats

# 生成示例数据
np.random.seed(0)
data = np.random.normal(loc=50, scale=10, size=1000)  # 正态分布数据
data = np.append(data, [1000, -1000])  # 添加一些明显的异常值
df = pd.DataFrame(data, columns=['value'])

# 1. 计算Z-Score并识别异常值
z_scores = stats.zscore(df['value'])
threshold = 3  # 设置Z-Score阈值
outliers = df[np.abs(z_scores) > threshold]

print(f"异常值数量: {len(outliers)}")
print(f"异常值:\n{outliers}")

# 2. 基于标准差的方法
mean = df['value'].mean()
std = df['value'].std()
threshold_std = 3  # 设置标准差阈值
outliers_std = df[(df['value'] > mean + threshold_std * std) | (df['value'] < mean - threshold_std * std)]

print("\n基于标准差方法识别的异常值:")
print(outliers_std)

from sklearn.cluster import KMeans

# 3. 基于聚类的方法（K-Means）
# 这里我们假设大部分数据属于一个簇，异常值属于其他簇或噪声
kmeans = KMeans(n_clusters=1, random_state=0).fit(df[['value']])  # 实际上，K=1可能不适合直接找异常值，这里仅为演示
# 更合理的方法是使用如DBSCAN或选择合理的K值后分析小簇或噪声点，但此处简化处理
# 由于K=1不直接产生异常值标识，我们改用一种启发式：假设大部分点紧密聚集，计算每个点到簇中心的距离
center = kmeans.cluster_centers_[0][0]
distances = np.abs(df['value'] - center)
threshold_distance = np.percentile(distances, 99)  # 使用99百分位数作为阈值（近似）
outliers_kmeans = df[distances > threshold_distance]

# 注意：上述K-Means处理仅为演示异常值识别思路，实际应用中需更合理选择K或采用其他聚类算法分析噪声/小簇
print("\n基于聚类方法（启发式）识别的异常值:")
print(outliers_kmeans)

from sklearn.cluster import DBSCAN

# 4. 基于密度的方法（DBSCAN）
dbscan = DBSCAN(eps=30, min_samples=5).fit(df[['value']])  # eps和min_samples需根据数据调整
labels = dbscan.labels_
outliers_dbscan = df[labels == -1]  # -1表示噪声点（异常值）

print("\n基于密度方法（DBSCAN）识别的异常值:")
print(outliers_dbscan)

from sklearn.ensemble import IsolationForest

# 5. 基于机器学习的方法（孤立森林）
iso_forest = IsolationForest(contamination=0.01, random_state=0).fit(df[['value']])  # contamination设置异常值比例
outliers_iso_forest = df[iso_forest.predict(df[['value']]) == -1]  # -1表示异常值

print("\n基于机器学习方法（孤立森林）识别的异常值:")
print(outliers_iso_forest)