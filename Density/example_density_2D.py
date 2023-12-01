import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


N = 20

#Question 5
#kd = np.random.rand(N, 2)

#Question 6
kd = np.random.normal(0, 1, size=[int(0.7*N), 2])

grid_size = 100
Gx = np.arange(0, 1, 1/grid_size)
Gy = np.arange(0, 1, 1/grid_size)
Gx, Gy = np.meshgrid(Gx, Gy)

bandwidth = 0.05

kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(kd)
Z = np.exp(kde.score_samples(np.hstack(((Gx.reshape(grid_size*grid_size))[:,np.newaxis], (Gy.reshape(grid_size*grid_size)[:,np.newaxis])))))


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(Gx, Gy, Z.reshape(grid_size, grid_size), rstride = 1, cstride = 1, cmap = cm.coolwarm, linewidth = 0, antialiased = True)
ax.scatter(kd[:,0], kd[:,1], -10)
plt.show()

#Question 5
#Comme attendu, l'estimation est moins restrictive sur les groupes si on augmente la bandwidth

#Question 6
#
