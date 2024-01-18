# importations préalables
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.impute import KNNImputer

# Génération des données bidimensionnelles complètes
n_base = 100
data1 = np.random.randn(n_base,2) + [5,5]
data2 = np.random.randn(n_base,2) + [3,2]
data3 = np.random.randn(n_base,2) + [1,5]
data = np.concatenate((data1,data2,data3))

np.random.shuffle(data)

# visualisation (optionnelle) des données générées
plt.plot(data[:,0],data[:,1],'r+')
#plt.show()

n_samples = data.shape[0]
missing_rate = 0.3

n_missing_samples = int(np.floor(n_samples * missing_rate))
print("Nous allons supprimer {} valeurs".format(n_missing_samples))

# choix des lignes à valeurs manquantes
present = np.zeros(n_samples - n_missing_samples, dtype=np.bool_)
missing = np.ones(n_missing_samples, dtype=np.bool_)

missing_samples = np.concatenate((present, missing))
# On mélange le tableau des valeurs absentes
np.random.shuffle(missing_samples)

# obtenir la matrice avec données manquantes : manque indiqué par True
data_missing = data.copy()
data_missing[np.where(missing_samples), 1] = np.nan

#----------------------------------------------------------------------------------------------

# Question 1
# le mécanisme qui représente les données manquantes est MCAR car manquantes de manièe totalement aléatoires

#----------------------------------------------------------------------------------------------

# imputation par la moyenne, la médiane puis par le remplacement de 0 quand donnée manquante
impMean = SimpleImputer(missing_values=np.nan, strategy='mean')
impMedian = SimpleImputer(missing_values=np.nan, strategy='median')
impZero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

data_imputed_mean = impMean.fit_transform(data_missing)
data_imputed_median = impMedian.fit_transform(data_missing)
data_imputed_zero = impZero.fit_transform(data_missing)

#plt.scatter(data_imputed_mean[:,0],data_imputed_mean[:,1], marker='+', c=missing_samples)
#plt.show()
#plt.scatter(data_imputed_median[:,0],data_imputed_median[:,1], marker='+', c=missing_samples)
#plt.show()
#plt.scatter(data_imputed_zero[:,0],data_imputed_zero[:,1], marker='+', c=missing_samples)
#plt.show()

# calculer l'"erreur" d'imputation
print("mean error : ", mean_squared_error(data[missing_samples,1],data_imputed_mean[missing_samples,1]))
print("median error : ", mean_squared_error(data[missing_samples,1],data_imputed_median[missing_samples,1]))
print("fill 0 error : ", mean_squared_error(data[missing_samples,1],data_imputed_zero[missing_samples,1]))

#----------------------------------------------------------------------------------------------

# Question 2
# afficher avec plt.show() ?

# Question 3
# l'erreur résultante est sensiblement la même que l'erreur par  moyenne

# Question 4
# l'erreur obtenue est nettement supérieure quand on rempli avec des zeros car on ne tiens pas compte des données, le remplissage est toalement indépendant

#----------------------------------------------------------------------------------------------

# Test de la précision en utilisant la moyenne par groupe avec kmeans

# obtenir le tableau composé des seules observations complètes
# ~ permet d'inverser un tableau de booléens
data_filtered = data[~missing_samples, :]

# application de la classification automatique aux observations complètes
kmeans = KMeans(n_clusters=3).fit(data_filtered)
# affichage des centres obtenus pour les groupes
centers = kmeans.cluster_centers_
#plt.scatter(data_filtered[:,0], data_filtered[:,1], marker='+', c=kmeans.labels_)
#plt.scatter(centers[:,0], centers[:,1], edgecolors='k', s=300, marker='*', label="Centres", c=range(len(centers)))
#plt.legend()
#plt.show()

y = np.array([1, 2, 3])    # les étiquettes des groupes
ncPredictor = NearestCentroid()

# les centres calculés par k-means sont associés aux 3 étiquettes des groupes
#  (les 'observations' pour NearestCentroid sont les centres issus de k-means)
#   seules les coordonnées des centres sur l'axe 1 sont employées
ncPredictor.fit(centers[:,0].reshape(-1, 1), y)

# l'index du centre le plus proche est déterminé pour chaque observation à
#  donnée manquante (à partir de la valeur de la variable non manquante, axe 1
nearest = ncPredictor.predict(data_missing[missing_samples, 0].reshape(-1,1))
# détermination des valeurs à utiliser pour l'imputation : pour chaque
#  observation, la valeur sur l'axe 2 du centre correspondant
estimated = np.zeros(n_missing_samples)
indices = range(n_missing_samples)
for i in indices:
    estimated[i] = centers[nearest[i]-1,1]

data_imputed = data_missing.copy()   # initialisation de data_imputed
# imputation avec les valeurs obtenues
data_imputed[missing_samples, 1] = estimated
# calcul de l'erreur moyenne d'imputation
print("k means error : ", mean_squared_error(data[missing_samples,1],data_imputed[missing_samples,1]))

#plt.scatter(data_imputed[:,0], data_imputed[:,1], marker='+', c=missing_samples)
#plt.scatter(centers[:,0], centers[:,1], edgecolors='k', s=300, marker='*', label="Centres", c='r')
#plt.legend()
#plt.show()

#----------------------------------------------------------------------------------------------

# Question 5
# avec des groupes plus séparés, la précision de l'imputation par moyenne ou par médiane ne serait pas meilleure, mais 
# l'imputation avec des groupes le serait. Nous pouvons le tester en remplaçant les valeurs des décalages dans 
# data1 = np.random.randn(n_base,2) + [5,5]
# data2 = np.random.randn(n_base,2) + [3,2]
# data3 = np.random.randn(n_base,2) + [1,5]
# par exemple : 
# data1 = np.random.randn(n_base,2) + [7,5]
# data2 = np.random.randn(n_base,2) + [4,2]
# data3 = np.random.randn(n_base,2) + [1,5]

#----------------------------------------------------------------------------------------------

# imputation sur la base des 3 plus proches voisins sans données manquantes
#  obtenus à partir de distances calculées avec les seules données présentes
data_imputed = KNNImputer(missing_values=np.nan, n_neighbors=3).fit_transform(data_missing)
# évaluation des résultats
print("k neighbours error : ", mean_squared_error(data[missing_samples,1],data_imputed[missing_samples,1]))
plt.scatter(data_imputed[:,0], data_imputed[:,1], marker='+', c=missing_samples)
#plt.show()

