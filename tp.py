#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:08:52 2017

@author: cazanave
"""

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
import matplotlib.pyplot as plt
from time import time
from collections import Counter

digits = load_digits()
data = digits.data

#####################################################################
#-------------------------------------------------------------------
# PART 1 : KMeans and Principal Component Analysis
#-------------------------------------------------------------------
#####################################################################

# first test of the kmeans algorithm with n_clusters=8
K=8
kmeans_model = KMeans(n_clusters=K).fit(data)
labels = kmeans_model.labels_
silhouette_score(data, labels, metric='euclidean')

# plot the first 4 images of the dataset, and their associated label
images = data.reshape((-1,8,8))
plt.suptitle('Images des quatre premières instances du dataset, et numéros de cluster associés')
for i in range(4):
    plt.subplot(141+i)
    plt.imshow(images[i],cmap=plt.cm.gray_r,interpolation="nearest")
    print(kmeans_model.labels_[i])
    plt.text(.99, .01, ('%d' % kmeans_model.labels_[i]),
                 transform=plt.gca().transAxes, size=15, color='r',
                 horizontalalignment='right')

# plot images of the centroids of the K clusters
img = kmeans_model.cluster_centers_.reshape((-1,8,8))
for i in range(K):
    plt.subplot(1+(K//5),5,1+i)
    plt.imshow(img[i],cmap=plt.cm.gray_r,interpolation="nearest")

# test of kmeans with init='random'
scores = []
for i in range(300):
    km = KMeans(init='random').fit(data)
    labels = km.labels_
    scores.append(silhouette_score(data, labels, metric='euclidean'))
print("# init='random'")
print("min : ", min(scores))
print("max : ", max(scores))
print("mean : ", sum(scores)/len(scores))

# test again kmeans with default parameter init='k-means++' to see the difference with init='random'
km = KMeans(init='k-means++').fit(data)
labels = km.labels_
print("# init='k-means++'")
print(silhouette_score(data, labels, metric='euclidean'))

#variation of n_clusters from 2 to 15, test of kmeans without ACP
scores = []
times = []
for k in range(2,16):
    t0 = time()
    kmeans_model = KMeans(n_clusters=k).fit(data)
    times.append(time()-t0)
    labels = kmeans_model.labels_
    scores.append(silhouette_score(data, labels, metric='euclidean'))
plt.title('Variation en fonction du nombre de clusters')
plt.xlabel('nombre de clusters')
x = range(2,16)
plt.plot(x, scores, label='silhouette', marker='x')
plt.legend()
plt.xlabel('nombre de clusters')
plt.plot(x, times, label='temps', marker='x')
plt.legend()
print("max : ", max(scores))

plt.figure()

#same varaiation of n_clusters parameter, test of kmeans after an ACP with n_components=8
scores = []
times = []
pca = PCA(n_components = 8)
pca.fit(data)
X = pca.transform(data)
for k in range(2,16):
    t0 = time()
    kmeans_model = KMeans(n_clusters=k).fit(X)
    times.append(time()-t0)
    labels = kmeans_model.labels_
    scores.append(silhouette_score(X, labels, metric='euclidean'))
plt.title('Variation en fonction du nombre de clusters')
plt.xlabel('nombre de clusters')
x = range(2,16)
plt.plot(x, scores, label='silhouette', marker='x')
plt.legend()
plt.xlabel('nombre de clusters')
plt.plot(x, times, label='temps', marker='x')
plt.legend()
print("max : ", max(scores))


#####################################################################
#-------------------------------------------------------------------
# PART 2 : Agglomerative Clustering
#-------------------------------------------------------------------
#####################################################################

# test of agglomerative clustering, variation of n_clusters from 2 to 15 with default linkage='ward'
# all this part is not necessary as it is included in the next test where we also change the linkage parameter
scores = []
times = []
for k in range(2,16):
    t0 = time()
    clustering_model = AgglomerativeClustering(n_clusters=k).fit(data)
    times.append(time()-t0)
    labels = clustering_model.labels_
    scores.append(silhouette_score(data, labels, metric='euclidean'))  
plt.title('Variation du score en fonction du nombre de clusters')
plt.xlabel('nombre de clusters')
x = range(2,16)
plt.ylabel('Score')
plt.plot(x, scores, label='silhouette', marker='x')
plt.legend()
plt.figure()
plt.title('Variation du temps en fontion du nombre de clusters')
plt.xlabel('nombre de clusters')
plt.ylabel('Temps')
plt.plot(x, times, label='temps', marker='x')
plt.legend()

# test of agglomerative clustering, variation of n_clusters from 2 to 15, and variation of linkage parameter
scores = [[]]
times = [[]]
linkages = ['ward','complete','average']
for linkage, i in zip(linkages, range(3)):
    print(i," ",linkage)
    for k in range(2,16):
        print(k)
        t0 = time()
        clustering_model = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(data)
        times[i].append(time()-t0)
        labels = clustering_model.labels_
        scores[i].append(silhouette_score(data, labels, metric='euclidean'))
    if i < len(linkages)-1:
        scores.append([])
        times.append([])
plt.subplot(121)       
plt.title('Variation du score en fonction du linkage')
plt.xlabel('k')
x = range(2,16)
plt.ylabel('Score')
for i in range(len(linkages)):
    plt.plot(x, scores[i], label=linkages[i], marker='x')
plt.legend()
plt.show()
plt.subplot(122)       
plt.title('Variation du temps d\'apprentissage en fonction du linkage')
plt.xlabel('k')
plt.ylabel('Temps d\'apprentissage')
for i in range(len(linkages)):
    plt.plot(x,times[i], label=linkages[i], marker='x')
plt.legend()
plt.show()


#####################################################################
#-------------------------------------------------------------------
# PART 3 : Complements
#-------------------------------------------------------------------
#####################################################################


##-------------------------------------------
## DBSCAN
##-------------------------------------------

# test of dbscan algorithm, variation of eps parameter
scores = []
times = []
n_clusters = []
epsilons = range(11,33)
for i in epsilons:
    print(i)
    t0 = time()
    db = DBSCAN(eps=i).fit(data)
    times.append(time()-t0)
    labels = db.labels_
    scores.append(silhouette_score(data, labels)  )
    n_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
    print(">>{0}\t{1}".format(n_clusters[-1:], scores[-1:]))
plt.subplot(131)
plt.title("Score en fonction d\'epsilon")    
plt.plot(epsilons, scores, 'r-', marker="x")
plt.xlabel("eps")
plt.ylabel("scores")
plt.subplot(132)
plt.title("Temps en fonction d\'epsilon")    
plt.plot(epsilons, times, 'g-', marker="x")
plt.xlabel("eps")
plt.ylabel("times")
plt.subplot(133)
plt.title("Nombre de clusters en fonction d\'epsilon")    
plt.plot(epsilons, n_clusters, 'b-', marker="x")
plt.xlabel("eps")
plt.ylabel("n_clusters")
plt.show()

# test of dbscan algorithm, variation of algorithm parameter
scores = []
times = []
n_clusters = []
algos = ['auto', 'ball_tree', 'kd_tree', 'brute']
for algo in algos:
    print(algo)
    t0 = time()
    db = DBSCAN(eps=25, algorithm=algo).fit(data)
    times.append(time()-t0)
    labels = db.labels_
    scores.append(calinski_harabaz_score(data, labels))
    n_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
x = range(len(algos))
plt.subplot(131)
plt.title("Score en fonction de l\'algorithme")    
plt.plot(x, scores, 'r-', marker="x")
plt.xlabel("algorihm")
plt.ylabel("scores")
plt.xticks(x,algos)
plt.subplot(132)
plt.title("Temps en fonction de l\'algorithme")    
plt.plot(x, times, 'g-', marker="x")
plt.xlabel("algorithm")
plt.ylabel("times")
plt.xticks(x,algos)
plt.subplot(133)
plt.title("Nombre de clusters en fonction de l\'algorithme")    
plt.plot(x, n_clusters, 'b-', marker="x")
plt.xlabel("algorithm")
plt.ylabel("n_clusters")
plt.xticks(x,algos)
plt.legend()
plt.show()





##-------------------------------------------
## Divisive clustering based on kmeans
##-------------------------------------------


def divisive_clustering(K):
    labels = [0] * len(data)
    for k in range(K-1):
        most_common = Counter(labels).most_common(1)[0][0]
        ii = [i for i, x in enumerate(labels) if x == most_common]
        new_data = [data[i] for i in ii]
        km = KMeans(n_clusters=2).fit(new_data)
        new_labels = km.labels_
        for i in range(len(new_labels)):
            if(new_labels[i] == 1):
                labels[ii[i]] = k+1
    print(K, ":\t", silhouette_score(data, labels, metric='euclidean'))

# test the divisive clustering algorithm
for k in range(2,16):
    divisive_clustering(k)









