# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:45:42 2017

@author: pierr
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

df_origin= pd.read_csv('consommation-electrique-par-secteurs-dactivite.csv', delimiter=';')
df_origin2= pd.read_csv('production-electrique-par-filiere.csv', delimiter=';')

tags = ['Code IRIS','Nb sites secteur résidentiel',
       'Conso totale secteur résidentiel (MWh)',
       'Conso moyenne secteur résidentiel (MWh)', 'Nb sites Agriculture',
       'Conso totale Agriculture (MWh)', 'Nb sites Industrie',
       'Conso totale Industrie (MWh)', 'Nb sites Tertiaire',
       'Conso totale Tertiaire (MWh)']
df_origin_group=df_origin.groupby('Code IRIS')
df_origin_group1=df_origin_group.get_group('100150000')

idx_last_yr = df_origin.groupby(['Code IRIS'], sort=False)['Année'].transform(max) == df_origin['Année']
df_origin_last_yr=df_origin[idx_last_yr]
df_origin_last_yr_trunc=df_origin_last_yr[tags]

df_origin_last_yr_trunc=df_origin_last_yr_trunc.set_index('Code IRIS')
df=df_origin_last_yr_trunc
df -= df.min()
df /= df.max()

df_woNan=df.fillna(0)


from pandas.plotting import scatter_matrix

#scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
#df_origin_last_yr_trunc_minmax = df_origin_last_yr_trunc.transform(df_origin_last_yr_trunc)
from sklearn.manifold import TSNE

nbrnanrow=df_woNan.isnull().sum(axis=1)
nbrnanrcol=df_woNan.isnull().sum(axis=0)
model = TSNE(n_components=2, random_state=0)

np.set_printoptions(suppress=True) 

nb_samples=1000
import random
random_idx=random.sample(range(len(df_woNan.index)), nb_samples)

ndarray_tsne=model.fit_transform(df_woNan.ix[random_idx]) 
df_tsne=pd.DataFrame(ndarray_tsne,columns=['a', 'b'])

df_tsne.plot.scatter(x='a', y='b');

from sklearn.cluster import AgglomerativeClustering
hierarch_cluster=AgglomerativeClustering(n_clusters=8)
cluster_nmb= np.zeros(shape=(nb_samples,1))
cluster_nmb=hierarch_cluster.fit_predict(df_tsne, y=cluster_nmb)

from matplotlib import pyplot as plt
# Visualize the clustering
def plot_clustering(X_red, y,labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)
    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
plot_clustering(df_tsne.as_matrix(), hierarch_cluster.labels_, hierarch_cluster.labels_, "Ward linkage")

#from matplotlib import pyplot
#pyplot.scatter(df_tsne.as_matrix()[:,0], df_tsne.as_matrix()[:,1], c=hierarch_cluster.labels_)
#pyplot.show()
df_woNan_selec=df_woNan.ix[random_idx]
df_woNan_selec['cluster']=hierarch_cluster.labels_
df_woNan_selec_group=df_woNan_selec.groupby('cluster')
df_woNan_selec_stats=df_woNan_selec_group.mean()
