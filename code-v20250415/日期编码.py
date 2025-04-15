import pandas as pd

# 3. 对日期进行编码
def date_to_timestamp_days(date_series):
    """
    将日期序列转换为以天为单位的时间戳（自1970年1月1日起）。

    参数:
    date_series (pd.Series): 包含日期的Pandas Series对象。

    返回:
    pd.Series: 包含时间戳（以天为单位）的Pandas Series对象。
    """

    # 确保输入是datetime类型

    if date_series.dtype == 'object':
        try:
            date_series_convert = pd.to_datetime(date_series, format='%Y-%m-%d')
        except Exception as e:
            print(f"转换为datetime类型时出错: {e}")
    else:
        print("'已经是非object类型，无需转换。")

    if not pd.api.types.is_datetime64_any_dtype(date_series_convert):
        raise ValueError("输入的Series必须是datetime类型")

    # 计算时间戳（以天为单位）
    timestamp_days = (date_series_convert - pd.Timestamp("1970-01-01")).dt.days

    return timestamp_days

X_train = pd.read_csv(r"D:\wk\myProject\project_insureFraudPredict\data\train.csv")
X_train['incident_date_timestamp_days'] = date_to_timestamp_days(X_train['incident_date'])
X_train['policy_bind_date_timestamp_days'] = date_to_timestamp_days(X_train['policy_bind_date']) 