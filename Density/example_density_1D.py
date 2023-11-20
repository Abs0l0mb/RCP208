import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

N = 100000

#génère 30% de points autour de 1, 70% de points autour de 5, selon une répartition gaussienne à chaque fois
array1 = np.random.normal(0, 1, int(0.3 * N))
array2 = np.random.normal(5, 1, int(0.7 * N))

X = np.concatenate([array1, array2])[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_density = (0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)

density = np.exp(kde.score_samples(X_plot))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill(X_plot[:, 0], true_density, fc='b', alpha=0.2, label='Vraie densité')
ax.plot(X_plot[:, 0], density, '-', label='Estimation')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
ax.legend(loc='upper left')
plt.show()

