import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

N = 100

array1 = np.random.normal(0, 1, int(0.1 * N))
array2 = np.random.normal(5, 1, int(0.4 * N))
array3 = np.random.normal(10, 1, int(0.3 * N))
array4 = np.random.normal(15, 1, int(0.2 * N))

X = np.concatenate([array1, array2, array3, array4])[:, np.newaxis]
X_plot = np.linspace(-5, 20, 1000)[:, np.newaxis]

true_density = (0.1 * norm(0, 1).pdf(X_plot[:, 0]) + 0.4 * norm(5, 1).pdf(X_plot[:, 0]) + 0.3 * norm(10, 1).pdf(X_plot[:, 0]) + 0.2 * norm(15, 1).pdf(X_plot[:, 0]))

#kde = KernelDensity(kernel='gaussian', bandwidth=0.25).fit(X)
kde = KernelDensity(kernel='linear', bandwidth=0.75).fit(X)
#kde = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(X)

density = np.exp(kde.score_samples(X_plot))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.fill(X_plot[:, 0], true_density, fc='b', alpha=0.2, label='Vraie densité')
ax.plot(X_plot[:, 0], density, '-', label='Estimation')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
ax.legend(loc='upper left')
plt.show()

#Question 2 : Faites varier le nombre de lois normales lors de la génération de l’échantillon et examinez les résultats.
#On observe que les groupes se confondent très vite, d'ou le fait qu'on les ait espacées de 5 au lieu de 3 initialement
#Cependant si on les espaces assez, l'algorithme de kernel density est capable de les séparer.
