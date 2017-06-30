# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:45:42 2017

@author: pierr
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

#IMPORTATION DES DONNEES
df_origin1= pd.read_csv('consommation-electrique-par-secteurs-dactivite.csv', delimiter=';')
#df_origin2= pd.read_csv('production-electrique-par-filiere.csv', delimiter=';')
df_origin3= pd.read_csv('HDE_surfaces_iris.csv', delimiter='\t')
#df_origin4= pd.read_csv('HDE_iris_nombre de personnes par iris.csv', delimiter='\t')

#INDEXATION PAR LE CODE IRIS
tags = ['Code IRIS','Nb sites secteur résidentiel',
       'Conso totale secteur résidentiel (MWh)',
       'Conso moyenne secteur résidentiel (MWh)', 'Nb sites Agriculture',
       'Conso totale Agriculture (MWh)', 'Nb sites Industrie',
       'Conso totale Industrie (MWh)', 'Nb sites Tertiaire',
       'Conso totale Tertiaire (MWh)']
df_origin=df_origin1
df_origin_group=df_origin.groupby(['INSEE IRIS'])
df_origin_group1=df_origin_group.get_group('100150000')


#LIMITATION A LA DERNIERE ANNEE
idx_last_yr = df_origin.groupby(['Code IRIS'], sort=False)['Année'].transform(max) == df_origin['Année']
df_origin_last_yr=df_origin[idx_last_yr]
#tags3=['INSEE IRIS','INSEE COMM', 'Nom commune']
df_origin_last_yr_trunc=df_origin_last_yr[tags]

df_origin_last_yr_trunc=df_origin_last_yr_trunc.set_index(['Code IRIS'])

#tags2=['INSEE IRIS','INSEE COMM', 'Nom commune', 'Code département', 'NbHbts','NbUniConso ']

#commontag=list(set(tags3).intersection(tags2))
#df_origin4short=df_origin4[tags2]

#result = pd.merge(df_origin_last_yr_trunc, df_origin4short, how='outer', on=commontag)
#df_origin4short=df_origin4short.set_index('INSEE IRIS')
#dftest=pd.concat([df_origin_last_yr_trunc, df_origin4short], axis=1)
#NORMALISATION A [0,1]
df=df_origin_last_yr_trunc
df -= df.min()
df /= df.max()

#REMPLISSAGE DES CASES VIDES PAR DES 0
df_woNan=df.fillna(0)


#from pandas.plotting import scatter_matrix
#scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
#df_origin_last_yr_trunc_minmax = df_origin_last_yr_trunc.transform(df_origin_last_yr_trunc)

#REDUCTION DE DIMENSION POUR VISUALISATION
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True) 

#LIMITATION DU NOMBRE DE SAMPLES
nb_samples=1000
import random
random_idx=random.sample(range(len(df_woNan.index)), nb_samples)
ndarray_tsne=model.fit_transform(df_woNan.ix[random_idx]) 

#PLOT EN 2D PRE-CLUSTER
df_tsne=pd.DataFrame(ndarray_tsne,columns=['a', 'b'])
df_tsne.plot.scatter(x='a', y='b');

#CLUSTERISATION HIERARCHIQUE
from sklearn.cluster import AgglomerativeClustering
hierarch_cluster=AgglomerativeClustering(n_clusters=8)
cluster_nmb= np.zeros(shape=(nb_samples,1))
cluster_nmb=hierarch_cluster.fit_predict(df_tsne, y=cluster_nmb)

#PLOT EN 2D POST-CLUSTER
from matplotlib import pyplot as plt
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
    plt.axis('on')
    plt.show()
plot_clustering(df_tsne.as_matrix(), hierarch_cluster.labels_, hierarch_cluster.labels_, "Ward linkage")

#PLOT EN 2D POST-CLUSTER ALTERNATIF (SANS NUMERO DE CLUSTER)
#from matplotlib import pyplot
#pyplot.scatter(df_tsne.as_matrix()[:,0], df_tsne.as_matrix()[:,1], c=hierarch_cluster.labels_)
#pyplot.show()

#AJOUT DU NUMERO DU CLUSTER
df_woNan_selec=df_woNan.ix[random_idx]
df_woNan_selec['cluster']=hierarch_cluster.labels_

#STATISTIQUES PAR CLUSTER
df_woNan_selec_group=df_woNan_selec.groupby('cluster')
df_woNan_selec_stats=df_woNan_selec_group.mean()


#from sklearn.svm import OneClassSVM
#outlier_detect_svc=OneClassSVM()
#outlier_detect_svc.fit(df_tsne)
#outliers_fraction = 0.25
#scores_pred = outlier_detect_svc.decision_function(df_tsne)
#y_pred = outlier_detect_svc.predict(df_tsne)
#
##PLOT EN 2D POST-CLUSTER ALTERNATIF (SANS NUMERO DE CLUSTER)
#from matplotlib import pyplot
#plt.figure()
#
#pyplot.scatter(df_tsne.as_matrix()[:,0], df_tsne.as_matrix()[:,1], c=y_pred)
#pyplot.show()