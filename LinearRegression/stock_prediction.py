# cross_validation 0.2开始废除
import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import quandl
from matplotlib import style
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
# 这是一个股票预测的例子
# 修改matplotlib样式
style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

# 定义预测列变量，它存放研究对象的标签名
forecast_col = 'Adj. Close'
# 定义预测天数，这里设置为所有数据量长度的1%
forecast_out = int(math.ceil(0.01 * len(df)))

# 只用到df中下面的几个字段
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# 构造两个新的列
# HL_PCT为股票最高价与最低价的变化百分比
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
# HL_PCT为股票收盘价与开盘价的变化百分比
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
# 下面为真正用到的特征字段
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# 因为scikit-learn并不会处理空数据，需要把为空的数据都设置为一个比较难出现的值，这里取-9999，
df.fillna(-99999, inplace=True)
# 用label代表该字段，是预测结果
# 通过让与Adj. Close列的数据往前移动1%行来表示
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# 抛弃label列中为空的那些行
df.dropna(inplace=True)
y = np.array(df['label'])
# 上面的这些东西是对数据的预处理，一堆巴拉巴拉的我们不需要知道
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# 分配训练集和测试集
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(acc, forecast_set)
#线性回归预测
one_day = 86400
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + one_day
# 遍历预测结果，用它往df追加行
# 这些行除了Forecast字段，其他都设为np.nan
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # 这段[np.nan for _ in range(len(df.columns) - 1)]代码生成不包含Forecast字段的列表
    # 而[i]为只包含Forecast值的列表
    # 上述两个列表拼接在一起就组成了新行，按日期追加到df的下面
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# 开始绘图
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
