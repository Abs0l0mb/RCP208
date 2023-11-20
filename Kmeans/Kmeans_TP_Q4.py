import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics

def get_inertia(n_cluster, data, labels, algorithm="random"):
    kmeans = KMeans(n_clusters=n_cluster, n_init=1, init=algorithm).fit(data)
    pred = kmeans.predict(data)
    #print(metrics.adjusted_rand_score(pred, labels))
    return kmeans.inertia_

def get_rand_score(n_cluster, data, labels, algorithm="random"):
    kmeans = KMeans(n_clusters=n_cluster, n_init=1, init=algorithm).fit(data)
    pred = kmeans.predict(data)
    #show_3D(data, pred)
    return metrics.adjusted_rand_score(pred, labels)

def show_3D(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels)
    plt.show()

data = np.random.uniform(low=0, high=1, size=(500, 3))

c1 = np.ones(100)
c2 = 2*np.ones(100)
c3 = 3*np.ones(100)
c4 = 4*np.ones(100)
c5 = 5*np.ones(100)
labels = np.concatenate((c1,c2,c3,c4,c5))

#show_3D(data, labels) #shows uniform distribution of point between [0, 0, 0] and [1, 1, 1] in 5 random groups 

data, labels = shuffle(data, labels)

rand_indexes_random = []
for i in range(1000):
    rand_indexes_random.append(get_rand_score(5, data, labels, "random"))
#print(rand_indexes_random) #rand index is close to 0
print(sum(rand_indexes_random)/len(rand_indexes_random)) #-0.0025, which means the clustering is especially discordant

rand_indexes_kmeanspp = []
for i in range(1000):
    rand_indexes_kmeanspp.append(get_rand_score(5, data, labels, "k-means++"))
#print(rand_indexes_kmeanspp) #rand index is close to 0
print(sum(rand_indexes_kmeanspp)/len(rand_indexes_kmeanspp)) #-0.0025, which means the clustering is especially discordant

