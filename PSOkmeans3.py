# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:10:03 2019

@author: chocolate
针对原始数据进行聚类 二维可视化看是否合适
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
import timeit

def kmeans_one(data, n_cluster, cen):
    data_num, data_dim = data.shape
    dist_mat = np.zeros((data_num, n_cluster))
    group_index = np.zeros((data_num))
#    计算距离
    for i in range (0, data_num):
        for j in range (0, n_cluster):
            dist_mat[i,j] = np.linalg.norm(data[i,:] - cen[j,:])
#    分配label
    for i in range (0, data_num):
        group_index[i] = np.where(dist_mat[i,:]==np.min(dist_mat[i,:]))[0][0]
    return group_index

def CHfitness(data, labels):
    score = metrics.calinski_harabaz_score(data, labels)
    return score

def PSOcenters(data, n_cluster, n_particle, kmax):
    data_num, data_dim = data.shape
    cen_arr = np.zeros((n_particle, n_cluster, data_dim))
    pbest_arr = np.zeros((n_particle, n_cluster, data_dim))
    fit_pbest_arr = np.zeros((n_particle))
    gbest = np.zeros((n_cluster, data_dim))
    fit_gbest = 0
    v_arr = cen_arr.copy()
#    给出常数
    omega = 0.5
    c1 = 2
    c2 = 2
#    随机初始化速度与粒子位置
#    cen_arr = np.random.uniform(low=-40, high=40, size = (n_particle, n_cluster, data_dim))
#    随机初始化效果不好 选择点进行初始化质心
    for i in range(0, n_particle):
        for j in range(0, n_cluster):
            random_index = np.random.randint(low = 0, high = data_num-1)
            cen_arr[i][j] = data[random_index]
#    print(cen_arr)
    v_arr = np.random.uniform(low=-1, high=1, size = (n_particle, n_cluster, data_dim))
    for k in range(0, kmax):
#        计算全体最优值和个体最优值
        for i in range(0, n_particle):
            labels = kmeans_one(data, n_cluster, cen_arr[i])
#            避免全部都归成一类
            if np.unique(labels).shape[0] == 1:
                labels[0] = 0
                labels[1] = 1
                labels[2] = 2
            temp = CHfitness(data, labels)
            if temp > fit_pbest_arr[i]:
                pbest_arr[i] = cen_arr[i]
            if temp > fit_gbest:
                gbest = cen_arr[i]
#        更新粒子
        for i in range(0, n_particle):
            r1 = np.random.rand()
            r2 = np.random.rand()
            v_arr[i] = omega * v_arr[i] + c1 * r1 *(pbest_arr[i] - cen_arr[i]) + c2 * r2 *(gbest - cen_arr[i])
            cen_arr[i] = v_arr[i] + cen_arr[i]
        k = k + 1
        print('%d'%(k/kmax*100),'%')
    return gbest,labels


start = timeit.default_timer()

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
alldata = df_fusion.values
#alldata 和 df_fusion 就是最后我们的得到融合完的特征 
#分别是ndarray和dataframe格式
# =============================================================================
#得到融合后的矩阵 直接聚类
alldata_kmeans = KMeans(n_clusters=3).fit(alldata)#直接聚类效果

#tsne联合pca降维
pca = PCA(n_components=80)
pca.fit(alldata)
alldata_pca = pca.transform(alldata)
tsne = TSNE(n_components=2, perplexity=10, learning_rate=150 , init='pca')
alldata_tsne = tsne.fit_transform(alldata_pca)
alldata_tsnekmeans = KMeans(n_clusters=3).fit(alldata_tsne)#联合聚类的效果

#绘制聚类图 利用df绘图方便
df = pd.DataFrame(alldata_tsne,index=alldata_tsnekmeans.labels_,columns=['x','y'])
df_k1 = df[df.index==0]
df_k2 = df[df.index==1]
df_k3 = df[df.index==2]
fig = plt.figure(figsize=(10, 10))
plt.scatter(df_k1.loc[:,['x']],df_k1.loc[:,['y']], s=50, c='red',marker='d')
plt.scatter(df_k2.loc[:,['x']],df_k2.loc[:,['y']], s=50, c='green',marker='*')
plt.scatter(df_k3.loc[:,['x']],df_k3.loc[:,['y']], s=50, c='brown',marker='p')
# =============================================================================

#PSO寻找质心
n_clusters = 3
n_particle = 50
kmax = 100

#20820维
best_cens, labels = PSOcenters(alldata, n_clusters, n_particle, kmax)
#2维
best_cens_2d, labels_2d = PSOcenters(alldata_tsne, n_clusters, n_particle, kmax)

score = metrics.calinski_harabaz_score(alldata_tsne, labels_2d)
#打印耗时间
end = timeit.default_timer()
print('-----------------%s'%(end-start))