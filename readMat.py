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
from sklearn import metrics

#class plotParameter:
    
def plotCluster(data, labels):
    df = pd.DataFrame(data, index = labels)
    df_k1 = df[df.index==0]
    df_k2 = df[df.index==1]
    df_k3 = df[df.index==2]
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(df_k1.iloc[:,[0]],df_k1.iloc[:,[1]], s=50, c='red',marker='d')
    plt.scatter(df_k2.iloc[:,[0]],df_k2.iloc[:,[1]], s=50, c='green',marker='*')
    plt.scatter(df_k3.iloc[:,[0]],df_k3.iloc[:,[1]], s=50, c='brown',marker='p')

def main():
#    读取到的是一个字典 所有的数据都在's'最后一项的key对应的
    s = scio.loadmat('S.mat')
    s = s['S']
#    s = pd.DataFrame(s)
    
    ydata = scio.loadmat('ydata.mat')
    ydata = ydata['ydata']
    
    ydata_kmeans = KMeans(n_clusters=3).fit(ydata)
    plotCluster(ydata, labels=ydata_kmeans.labels_)
    score = metrics.calinski_harabaz_score(ydata, ydata_kmeans.labels_)

if __name__=='__main__':
    main()







