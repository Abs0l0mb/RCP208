import numpy as np
from sklearn.naive_bayes import GaussianNB

# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(42)

# 1. Génération de 1000 échantillons de la densité N(10, 2)
mu_true = 10
sigma_true = np.sqrt(2)
sample_size = 1000

data_sample = np.random.normal(mu_true, sigma_true, sample_size)

# 2. Estimation des paramètres μ et σ^2 en utilisant scikit-learn
data_sample_reshaped = data_sample.reshape(-1, 1)  # Reshape pour s'adapter au modèle

# Créer et entraîner le modèle Gaussian Naive Bayes
model = GaussianNB()
model.fit(data_sample_reshaped, np.zeros_like(data_sample_reshaped))  # Les étiquettes ne sont pas utilisées dans ce contexte

# Extraire les paramètres estimés du modèle
estimated_mu = model.theta_[0][0]
estimated_sigma = np.sqrt(model.sigma_[0][0])

# Afficher les résultats
print(f"Paramètres réels   : μ = {mu_true}, σ^2 = {sigma_true}")
print(f"Paramètres estimés : μ = {estimated_mu}, σ^2 = {estimated_sigma}")