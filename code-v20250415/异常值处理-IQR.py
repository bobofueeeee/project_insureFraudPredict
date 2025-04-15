import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
data = np.random.normal(loc=50, scale=10, size=1000)  # 正态分布数据
data = np.append(data, [1000, -1000])  # 添加一些明显的异常值
df = pd.DataFrame(data, columns=['value'])


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # 下四分位数
    Q3 = df[column].quantile(0.75)  # 上四分位数
    IQR = Q3 - Q1  # 四分位距

    # 定义异常值的上下界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 识别异常值
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return outliers, lower_bound, upper_bound


# 识别异常值
outliers, lower_bound, upper_bound = detect_outliers_iqr(df, 'value')

print(f"异常值数量: {len(outliers)}")
print(f"异常值:\n{outliers}")
print(f"下界: {lower_bound}, 上界: {upper_bound}")

# 可视化数据和异常值
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['value'], label='data point', alpha=0.5)
plt.scatter(outliers.index, outliers['value'], color='red', label='unnormal value')
plt.axhline(y=lower_bound, color='green', linestyle='--', label='upper limit')
plt.axhline(y=upper_bound, color='green', linestyle='--', label='lower limit')
plt.title('unnormal value - IQR')
plt.xlabel('index')
plt.ylabel('value')
plt.legend()
plt.show()