import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 根据均值、标准差,求指定范围的正态分布概率值
def normfun(x, mu, sigma):
  pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
  return pdf


# result = np.random.randint(-65, 80, size=100) # 最小值,最大值,数量
result = np.random.normal(15, 44, 100) # 均值为0.5,方差为1
print(result)

x = np.arange(min(result), max(result), 0.1)
# 设定 y 轴，载入刚才的正态分布函数
print(result.mean(), result.std())
y = normfun(x, result.mean(), result.std())
plt.plot(x, y) # 这里画出理论的正态分布概率曲线

# 这里画出实际的参数概率与取值关系
plt.hist(result, bins=10, rwidth=0.8, density=True) # bins个柱状图,宽度是rwidth(0~1),=1没有缝隙
plt.title('distribution')
plt.xlabel('temperature')
plt.ylabel('probability')
# 输出
plt.show() # 最后图片的概率和不为1是因为正态分布是从负无穷到正无穷,这里指截取了数据最小值到最大值的分布