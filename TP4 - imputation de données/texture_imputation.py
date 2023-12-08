# importations préalables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.impute import KNNImputer

# Question 6 :

# load the data
raw_data = pd.read_csv("./texture.dat", sep='\s+', header=None)

# format it to array for processing
full_data = np.array(raw_data)

# get only the first two columns
data = full_data[:,:2]

#plt.plot(data[:,0],data[:,1],'r+')
#plt.show()

n_samples = data.shape[0]
missing_rate = 0.25
n_missing_samples = int(np.floor(n_samples * missing_rate))
print(n_missing_samples) #1375

present = np.zeros(n_samples - n_missing_samples, dtype=np.bool_)
missing = np.ones(n_missing_samples, dtype=np.bool_)

missing_samples = np.concatenate((present, missing))
# On mélange le tableau des valeurs absentes
np.random.shuffle(missing_samples)

# obtenir la matrice avec données manquantes : manque indiqué par True
data_missing = data.copy()
data_missing[np.where(missing_samples), 0] = np.nan

# imputation des données avec moyenne et médiane
impMean = SimpleImputer(missing_values=np.nan, strategy='mean')
impMedian = SimpleImputer(missing_values=np.nan, strategy='median')
data_imputed_mean = impMean.fit_transform(data_missing)
data_imputed_median = impMedian.fit_transform(data_missing)

# affichage des données imputées
#plt.scatter(data_imputed_mean[:,0],data_imputed_mean[:,1], marker='+', c=missing_samples)
#plt.show()
#plt.scatter(data_imputed_median[:,0],data_imputed_median[:,1], marker='+', c=missing_samples)
#plt.show()

# calculer les erreurs de prédiction
print("mean error : ", mean_squared_error(data[missing_samples,0],data_imputed_mean[missing_samples,0]))
print("median error : ", mean_squared_error(data[missing_samples,0],data_imputed_median[missing_samples,0]))

#----------------------------------------------------------------------------------------------

# Question 7
# On observe que l'imputation avec médiane est moins efficace que celle avec moyenne, cela est dû au fait que
# la répartition des données est très asymétrique.

# Question 8
# Appliquer kmeans à cet ensemble de variable ne me seble pas pertinent. Avec les informations à dispositions, il ne semble pas que les 11 classes représentées
# aient un impact sur les 2 premières variables, nous ne pourrions donc pas fixer le nombre de centre à 11 et espérer des résultats pertinents.
# Cependant peut être que l'algorithme des plus proches voisins peut être quelque chose d'intérressant à tester.
# Vérifions en appliquant et en observant l'erreur quadratique de chaque méthode :

#----------------------------------------------------------------------------------------------

# Test de la précision en utilisant la moyenne par groupe avec kmeans

data_filtered = data[~missing_samples, :]
kmeans = KMeans(n_clusters=11).fit(data_filtered)
centers = kmeans.cluster_centers_

plt.scatter(data_filtered[:,0], data_filtered[:,1], marker='+', c=kmeans.labels_)
plt.scatter(centers[:,0], centers[:,1], edgecolors='k', s=300, marker='*', label="Centres", c=range(len(centers)))
plt.legend()
plt.show()

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
ncPredictor = NearestCentroid()

ncPredictor.fit(centers[:,0].reshape(-1, 1), y)
nearest = ncPredictor.predict(data_missing[missing_samples, 1].reshape(-1,1))

estimated = np.zeros(n_missing_samples)
indices = range(n_missing_samples)
for i in indices:
    estimated[i] = centers[nearest[i]-1,1]

data_imputed = data_missing.copy()   # initialisation de data_imputed
# imputation avec les valeurs obtenues
data_imputed[missing_samples, 1] = estimated
# calcul de l'erreur moyenne d'imputation
print("k means error : ", mean_squared_error(data[missing_samples,1],data_imputed[missing_samples,1]))

plt.scatter(data_imputed[:,0], data_imputed[:,1], marker='+', c=missing_samples)
plt.scatter(centers[:,0], centers[:,1], edgecolors='k', s=300, marker='*', label="Centres", c='r')
plt.legend()
plt.show()

# imputation sur la base des 3 plus proches voisins sans données manquantes
#  obtenus à partir de distances calculées avec les seules données présentes
data_imputed = KNNImputer(missing_values=np.nan, n_neighbors=3).fit_transform(data_missing)
# évaluation des résultats
print("k neighbours error : ", mean_squared_error(data[missing_samples,1],data_imputed[missing_samples,1]))
plt.scatter(data_imputed[:,0], data_imputed[:,1], marker='+', c=missing_samples)
#plt.show()



DSADSA