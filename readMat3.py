# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:08:05 2019

@author: czpin
读取数据并且进行可视化操作
分析哪一种可视化的效果最好
"""
import pandas as pd
import scipy.io as scio
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans
from sklearn import metrics


n_feature = 4
alldata1 = scio.loadmat('alldata1.mat')
alldata1 = alldata1['alldata1']
alldata2 = scio.loadmat('alldata2.mat')
alldata2 = alldata2['alldata2']
alldata3 = scio.loadmat('alldata3.mat')
alldata3 = alldata3['alldata3']
alldata4 = scio.loadmat('alldata4.mat')
alldata4 = alldata4['alldata4']
df1 = pd.DataFrame(alldata1)
df2 = pd.DataFrame(alldata2)
df3 = pd.DataFrame(alldata3)
df4 = pd.DataFrame(alldata4)
df_fusion = df1.copy()

temparray = np.zeros((df_fusion.size, n_feature))
tempdf = pd.DataFrame(temparray)
temparray = np.zeros(df_fusion.size)
tempdf1d = pd.DataFrame(temparray)
alldata11d = df1.values.flatten()
alldata21d = df2.values.flatten()
alldata31d = df3.values.flatten()
alldata41d = df4.values.flatten()

#利用pca进行四个特征融合成一个特征
tempdf.iloc[:,0] = alldata11d
tempdf.iloc[:,1] = alldata21d
tempdf.iloc[:,2] = alldata31d
tempdf.iloc[:,3] = alldata41d
r, c = tempdf.shape
pca = PCA(n_components=1) 
pca.fit(tempdf)
df_fusion = pca.transform(tempdf)
df_fusion = df_fusion.reshape(alldata1.shape)
df_fusion = pd.DataFrame(df_fusion)

#得到融合后的矩阵 直接聚类
alldata_kmeans = KMeans(n_clusters=3).fit(df_fusion)#直接聚类效果
print(alldata_kmeans.labels_)

#pca降维
pca = PCA(n_components=2)
pca.fit(df_fusion)
alldata_pca = pca.transform(df_fusion)
fig = plt.figure(figsize=(10, 10))
plt.scatter(alldata_pca[:,0],alldata_pca[:,1])
plt.show()
alldata_kmeans = KMeans(n_clusters=3).fit(alldata_pca)#pca聚类的效果
print(alldata_kmeans.labels_)

#tsne降维
tsne = TSNE(n_components=2, perplexity=10, learning_rate=150 , init='pca')
alldata_tsne = tsne.fit_transform(df_fusion)
fig = plt.figure(figsize=(10, 10))
plt.scatter(alldata_tsne[:,0],alldata_tsne[:,1])
plt.show()
alldata_kmeans = KMeans(n_clusters=3).fit(alldata_tsne)#tsne聚类的效果
print(alldata_kmeans.labels_)

#tsne联合pca降维
pca = PCA(n_components=80)
pca.fit(df_fusion)
alldata_pca = pca.transform(df_fusion)
tsne = TSNE(n_components=2, perplexity=10, learning_rate=150 , init='pca')
alldata_tsne = tsne.fit_transform(alldata_pca)
fig = plt.figure(figsize=(10, 10))
plt.scatter(alldata_tsne[:,0],alldata_tsne[:,1])
plt.show()
alldata_kmeans = KMeans(n_clusters=3).fit(alldata_tsne)#联合聚类的效果
print(alldata_kmeans.labels_)

#绘制聚类图
df = pd.DataFrame(alldata_tsne,index=alldata_kmeans.labels_,columns=['x','y'])
df_k1 = df[df.index==0]
df_k2 = df[df.index==1]
df_k3 = df[df.index==2]
fig = plt.figure(figsize=(10, 10))
plt.scatter(df_k1.loc[:,['x']],df_k1.loc[:,['y']], s=50, c='red',marker='d')
plt.scatter(df_k2.loc[:,['x']],df_k2.loc[:,['y']], s=50, c='green',marker='*')
plt.scatter(df_k3.loc[:,['x']],df_k3.loc[:,['y']], s=50, c='brown',marker='p')


n_clusters = 3
n_particle = 50
kmax = 20
data_num, data_dim = df.shape
cen_arr = np.zeros((n_particle, n_clusters, data_dim))
alldata_kmeans = KMeans(n_clusters=3, init=cen_arr[0], n_init=1).fit(alldata_tsne)#n_init=1默认是10 必须改成1





