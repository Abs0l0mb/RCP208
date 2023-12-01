import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics

def get_inertia(n_cluster, data, labels, algorithm="random"):
    kmeans = KMeans(n_clusters=n_cluster, n_init=1, init=algorithm).fit(data)
    pred = kmeans.predict(data)
    print(metrics.adjusted_rand_score(pred, labels))
    return kmeans.inertia_

def show_3D(data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels)
    plt.show()

d1 = np.random.randn(100, 3) + [3,3,3]
d2 = np.random.randn(100, 3) + [-3,-3,-3]
d3 = np.random.randn(100, 3) + [-3,3,3]
d4 = np.random.randn(100, 3) + [-3,-3,3]
d5 = np.random.randn(100, 3) + [3,3,-3]
c1 = np.ones(100)
c2 = 2*np.ones(100)
c3 = 3*np.ones(100)
c4 = 4*np.ones(100)
c5 = 5*np.ones(100)

data = np.concatenate((d1,d2,d3,d4,d5))
labels = np.concatenate((c1,c2,c3,c4,c5))
data, labels = shuffle(data, labels)

#show_3D(data, labels)

points = []
for i in range(2, 20):
    points.append([i, get_inertia(i, data, labels, "random")])
    
arrayPoints = np.array(points)
x, y = arrayPoints.T
fig = plt.figure()
plt.scatter(x, y)
plt.show()

#Question 1
#on observe que la méthode d'initalisation random peut donner lieu à des adjusted rand score plus bas (min observé ~0.75)
#la méthode k-means++ donne au minimum des résultats aux alentours de 0.95

#Question 2
#en utilisant k-means++ on observe des résultats moins stables. Vu que le nombre de clusters est plus grand le nombre de
#groupes logiques, cela doit perturber l'initisalisation de k-means++, qui essaye de trouver une cohérence là ou il n'y
#en a pas forcément

#Question 3
#voir images fournies avec le repo

#Question 4
#
#

#Question 5
#
#

#Question 6
#
#

