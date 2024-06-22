from sklearn.metrics import confusion_matrix

# 假设你有一些真实的标签和模型预测的标签
y_true = [0, 1, 1, 0, 1, 1]  # 真实标签
y_pred = [0, 0, 1, 0, 0, 1]  # 模型预测的标签

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 打印混淆矩阵
print("混淆矩阵:")
print(cm)

# 从混淆矩阵中读取FP值
# 注意：在混淆矩阵中，FP是第二行第一列的值（索引为[1, 0]）
fp = cm[1, 0]

# 打印FP值
print("假正例（FP）的值:", fp)