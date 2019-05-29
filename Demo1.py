# -*- coding:utf-8 -*-
# 特征降维：主成分分析(PCA)  一、重构了有效的低维度特征向量，为数据展现提供了可能

import pandas as pd
import numpy as np

# 使用pandas分别读取训练数据与测试数据集。
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

#从训练集与测试集分离出64维度的像素特征与1维度的数字目标
#np.arange(64)表示[0 1 2 3 4.......63]
X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

from sklearn.decomposition import PCA
#初始化一个可以将高维度特征向量压缩至二维的PCA
estimator=PCA(n_components=2)
#利用estimator拟合测试集数据，并将结果存储在X_pca里
X_pca=estimator.fit_transform(X_digits)

#
from matplotlib import pyplot as plt
def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'green', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = X_pca[:, 0][y_digits.values == i]
        py = X_pca[:, 1][y_digits.values == i]
        plt.scatter(px, py, c=colors[i])
    
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
    
plot_pca_scatter()

# 《使用原始像素特征和经过PCA压缩重建的低维特征，在相同配置的支持向量机模型上进行图像识别》

# 对训练数据、测试数据进行特征向量（图片像素）与分类目标的分隔。
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 导入基于线性核的支持向量机分类器。
from sklearn.svm import LinearSVC

# 使用默认配置初始化LinearSVC，对原始64维像素特征的训练数据进行建模，并在测试数据上做出预测，存储在y_predict中。
svc = LinearSVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)

# 使用PCA将原64维的图像数据压缩到20个维度。
estimator = PCA(n_components=20)

# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征。
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转化（transform）。
pca_X_test = estimator.transform(X_test)

# 使用默认配置初始化LinearSVC，对压缩过后的20维特征的训练数据进行建模，并在测试数据上做出预测，存储在pca_y_predict中。
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_predict = pca_svc.predict(pca_X_test)

##  原始像素特征和经过PCA压缩重建的低维特征，在相同配置的支持向量机模型上识别性能的差异
from sklearn.metrics import classification_report
print(svc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=np.arange(10).astype(str))) 
print(pca_svc.score(pca_X_test,y_test))
print(classification_report(y_test,pca_y_predict,target_names=np.arange(10).astype(str))) 
