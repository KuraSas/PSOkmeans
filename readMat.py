# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:41:06 2019

@author: chocolate
"""
import pandas as pd
import scipy.io as scio
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans
data = scio.loadmat('gliomas_multi_omic_data.mat')
#读取到的是一个字典 所有的数据都在'g阿文liomas_multi_omic_data'最后一项
data = data['gliomas_multi_omic_data']
data2 = data[0][0]

s = scio.loadmat('S.mat')
s = s['S']
s = pd.DataFrame(s)

ydata = scio.loadmat('ydata.mat')
ydata = ydata['ydata']
ydata = pd.DataFrame(ydata)

fig = plt.figure(figsize=(12, 12)) # 设置画面大小
sns.heatmap(s,square=True)
plt.show()
fig = plt.figure(figsize=(12, 12))
plt.scatter(ydata[0],ydata[1])
plt.show()

#pca = PCA(n_components=2) 
#pca.fit(s)
#s_pca = pca.transform(s)
#fig = plt.figure(figsize=(12, 12))
#plt.scatter(s_pca[:,0],s_pca[:,1])
#plt.show()

fig = plt.figure()
pca = PCA(n_components = 80) 
pca.fit(s)
s_pca = pca.transform(s)
ax = Axes3D(fig)
ax.scatter(s_pca[:,0], s_pca[:,1], s_pca[:,2])

svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(s) 
s_pca = pca.transform(s)
ax = Axes3D(fig)
ax.scatter(s_pca[:,0], s_pca[:,1], s_pca[:,2])


fig = plt.figure()
tsne = TSNE(n_components=2, init='pca', random_state=501)
s_tsne = tsne.fit_transform(s)
fig = plt.figure(figsize=(12, 12))
plt.scatter(s_tsne[:,0],s_tsne[:,1])
plt.show()

fig = plt.figure()
tsne = TSNE(n_components=2)
s_tsne = tsne.fit_transform(s)
s_tsne_kmeans = KMeans(n_clusters=3, random_state=0).fit(s_tsne)
print(s_tsne_kmeans.labels_)
ax = Axes3D(fig)
ax.scatter(s_tsne[:,0], s_tsne[:,1])

fig = plt.figure()
fig = plt.figure(figsize=(12, 12))
plt.scatter(s_tsne[:,0],s_tsne[:,1])
plt.show()