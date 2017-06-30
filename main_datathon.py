# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:45:42 2017

@author: pierr
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

#IMPORTATION DES DONNEES
df_origin= pd.read_csv('all_data_v1_1_infos_iris.csv', delimiter=',')
df_origin_position= df_origin[['lon','lat']]
#INDEXATION PAR LE CODE IRIS
tags = ['id_iris','c_elec_conso_tot_res',
       'c_elec_conso_tot_agr', 'c_elec_conso_tot_indus',
       'c_elec_conso_tot_tert', 'p_elec_prod_pv',
       'p_elec_prod_eol','p_elec_prod_bio','p_elec_prod_coge']
df_origin_group=df_origin.groupby(['id_iris'])
#df_origin_group1=df_origin_group.get_group('100150000')

tags2 = ['c_elec_conso_tot_res',
       'c_elec_conso_tot_agr', 'c_elec_conso_tot_indus',
       'c_elec_conso_tot_tert', 'p_elec_prod_pv',
       'p_elec_prod_eol','p_elec_prod_bio','p_elec_prod_coge']
tags3=[s+'_par_hbts' for s in tags2]
test=df_origin[tags2].values/df_origin[['socioeco_npers']].values
df_origin[tags3]=pd.DataFrame(test)

tags3.append('id_iris')
df_origin=df_origin[tags3]
df_origin=df_origin.set_index(['id_iris'])
#REMPLISSAGE DES CASES VIDES PAR DES 0
df_woNan=df_origin.fillna(0)

#REDUCTION DE DIMENSION POUR VISUALISATION
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True) 

import time
start_time = time.time()
#LIMITATION DU NOMBRE DE SAMPLES
nb_samples=1000#len(df_woNan.index)
nb_clusters=10

import random
random_idx=random.sample(range(len(df_woNan.index)), nb_samples)
#random_idx=np.array(range(len(df_woNan.index)))

#CLUSTERISATION HIERARCHIQUE
from sklearn.cluster import AgglomerativeClustering
hierarch_cluster=AgglomerativeClustering(n_clusters=nb_clusters)
cluster_nmb= np.zeros(shape=(nb_samples,1))
cluster_nmb=hierarch_cluster.fit_predict(df_woNan.iloc[random_idx], y=cluster_nmb)
print("--- %s seconds ---" % (time.time() - start_time))
#
ndarray_tsne=model.fit_transform(df_woNan.iloc[random_idx]) 
##PLOT EN 2D PRE-CLUSTER
df_tsne=pd.DataFrame(ndarray_tsne,columns=['a', 'b'])
df_tsne.plot.scatter(x='a', y='b');
#
##PLOT EN 2D POST-CLUSTER
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
plot_clustering(df_tsne.as_matrix(), hierarch_cluster.labels_, hierarch_cluster.labels_, "Clusters énergétiques en France")

df_origin_position_small=df_origin_position.ix[random_idx]
plot_clustering(df_origin_position_small.as_matrix(), cluster_nmb,cluster_nmb, title=None)
#pyplot.scatter(df_origin_position[['lon']].ix[random_idx].as_matrix(), df_origin_position[['lat']].ix[random_idx].as_matrix(), c=hierarch_cluster.labels_)
#pyplot.show()

#AJOUT DU NUMERO DU CLUSTER
df_woNan_selec=df_woNan.ix[random_idx]
df_woNan_selec['cluster']=hierarch_cluster.labels_

#STATISTIQUES PAR CLUSTER
df_woNan_selec_group=df_woNan_selec.groupby('cluster')
df_woNan_selec_stats=df_woNan_selec_group.mean()

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(df_woNan)
distances, indices = nbrs.kneighbors(df_woNan)


#indicestest=pd.DataFrame(indices).loc[:1000].applymap(lambda x:df_positionIRIS.ix[x,['INSEE COMM']].valu)


#output=df_woNan[['id_iris']]
#output['Classe']=cluster_nmb
#output.to_csv('IRIS_10classes_ecoseulmnt.csv', sep=';', index=False)
