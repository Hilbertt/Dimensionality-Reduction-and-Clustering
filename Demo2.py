# -*- coding:utf-8 -*- 
# 无监督学习着重于发现数据本身的分布特点，无监督学习不需要对数据进行标记
# 无监督学习可以帮助我们发现数据的群落(聚类)，也可以降维，保留低维度特征

# 数据聚类是无监督学习的主流应用，最为经典且易用的聚类模型是：(K-means)K均值算法
# K-means：设定聚类的个数，迭代，更新聚类中心，再迭代，让数据点到聚类中心距离的平方和趋于稳定

# 分别导入numpy、matplotlib以及pandas，用于数学运算、作图以及数据分析。
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 使用pandas分别读取训练数据与测试数据集。
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

#从训练集与测试集分离出64维度的像素特征与1维度的数字目标
#np.arange(64)表示[0 1 2 3 4.......63]
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 从sklearn.cluster中导入KMeans模型。
from sklearn.cluster import KMeans

# 初始化KMeans模型，并设置聚类中心数量为10。
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)

# 逐条判断每个测试图像所属的聚类中心。
y_pred = kmeans.predict(X_test)

#性能评测方法 1、如果数据带有正确的类别信息，使用ARI，ARI指标与分类问题中的计算准确性的方法类似
# 从sklearn导入度量函数库metrics。
from sklearn import metrics
# 使用ARI进行KMeans聚类性能评估。
print ('使用API进行KMeans聚类性能评估的值是：',metrics.adjusted_rand_score(y_test, y_pred))

#性能评测方法 2、如果数据没有所属类别，我们用轮廓系数(Silhouette Coefficient)度量聚类结果的质量
#这里用一组简单的数据进行分析
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#分割出6个子图，并在1号作图
plt.subplot(3,2,1)

# 初始化原始数据点。
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# 在1号子图做出原始数据点阵的分布。
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)

# 绘制轮廓系数与不同类簇数量的直观显示图。
    plt.title('K = %s, silhouette coefficient= %0.03f' %(t, sc_score))
    
# 绘制轮廓系数与不同类簇数量的关系曲线。
plt.subplots_adjust(wspace =1, hspace =1)#调整子图间距
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')

plt.show()

#肘部观察法示例
# 导入必要的工具包。
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 使用均匀分布函数随机三个簇，每个簇周围10个数据样本。
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))

# 绘制30个数据样本的分布图像。
X = np.hstack((cluster1, cluster2, cluster3)).T    #X表示30组二维数据
plt.scatter(X[:,0], X[:, 1])    #将X拆分为两组，进行作图
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 测试9种不同聚类中心数量下，每种情况的聚类质量，并作图。
K = range(1, 10)
meandistortions = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])
    
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()

