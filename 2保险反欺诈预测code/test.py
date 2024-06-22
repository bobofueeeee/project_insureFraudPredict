import pandas as pd

# 创建一个简单的DataFrame
df = pd.DataFrame({
    'A': ['foo', 'bar', 'baz', 'foo', 'qux'],
    'B': ['one', 'one', 'two', 'three', 'two'],
    'C': [1, 2, 3, 4, 5],
    'D': [10, 20, 30, 20, 15]
})
print(df)

# 创建一个字典用于映射
mapping = {'foo': 'apple', 'bar': 'banana', 'baz': 'bat', 'qux': 'quail'}

# 使用map()将列A中的值替换为字典中的值
df['A'] = df['A'].map(mapping)

print(df)