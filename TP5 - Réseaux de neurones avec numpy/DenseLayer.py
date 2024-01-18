import numpy as np
import math
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn import datasets

class DenseLayer:
    def __init__(self, input_size, output_size, activation):
      self.input_size = input_size
      self.output_size = output_size
      self.activation = activation
      self.cache = None  # Le cache sera mis à jour lors de la passe forward
      self.Wxy = np.random.uniform(-(np.sqrt(6.0 / (input_size + output_size))), (np.sqrt(6.0 / (input_size + output_size))), size=(input_size, output_size))
      self.by = np.zero(output_size, 1)

    def dense_layer_forward(x, Wxy, by, activation):
        """
        Réalise une unique étape forward de la couche dense telle que décrite dans la figure précédente

        Arguments:
        x -- l'entrée, tableau numpy de dimension (n_x, m).
        Wxy -- Matrice de poids multipliant l'entrée, tableau numpy de shape (n_y, n_x)
        by -- Biais additif ajouté à la sortie, tableau numpy de dimension (n_y, 1)
        activation -- Chaîne de caractère désignant la fonction d'activation choisie : 'linear', 'sigmoid' ou 'relu'

        Retourne :
        y_pred -- prédiction, tableau numpy de dimension (n_y, m)
        cache -- tuple des valeurs utiles pour la passe backward (rétropropagation du gradient), contient (x, z)
        """    
        ### A COMPLETER  
        # calcul de z
        z = np.dot(Wxy,x) + by
        # calcul de la sortie en appliquant la fonction d'activation
        if activation == 'relu':
            y_pred = np.maximum(np.zeros(np.shape(z)),z)
        elif activation == 'sigmoid':
            y_pred = 1/(1+np.exp(-z))
        elif activation == 'linear':
            y_pred = z
        else:
            print("Erreur : la fonction d'activation n'est pas implémentée.")
        
        # sauvegarde du cache pour la passe backward
        cache = (x, z)
        
        return y_pred, cache

    def dense_layer_backward(dy_hat, Wxy, by, activation, cache):
        """
        Implémente la passe backward de la couche dense.

        Arguments :
        dy_hat -- Gradient de la fonction objectif par rapport à la sortie ŷ, de dimension (n_y, m)
        Wxy -- Matrice de poids multipliant l'entrée, tableau numpy de shape (n_y, n_x)
        by -- Biais additif ajouté à la sortie, tableau numpy de dimension (n_y, 1)
        cache -- dictionnaire python contenant des variables utiles (issu de dense_layer_forward())

        Retourne :
        gradients -- dictionnaire python contenant les gradients suivants :
                            dx -- Gradient de la fonction objectif par rapport aux entrées, de dimension (n_x, m)
                            dby -- Gradient de la fonction objectif par rapport aux biais, de dimension (n_y, 1)
                            dWxy -- Gradient de la fonction objectif par rapport aux poids synaptiques Wxy, de dimension (n_y, n_x)
        """
        
        # Récupérer les informations du cache
        (x, z) = cache
        
        # calcul de la sortie en appliquant l'activation
        if activation == 'relu':
            dyhat_dz = (z > 0).astype(int) 
        elif activation == 'sigmoid':
            dyhat_dz = (1/(1+np.exp(-z))) * (1 - (1/(1+np.exp(-z)))) #o(x) * (1 - o(x))
        elif activation == 'linear':
            dyhat_dz = np.ones(np.shape(z))
        else:
            print("Erreur : la fonction d'activation n'est pas implémentée.")

        # calculer le gradient de la perte par rapport à x
        dx = np.dot(np.transpose(Wxy), dy_hat*dyhat_dz)

        # calculer le gradient de la perte par rapport à Wxy
        dWxy = np.dot(dy_hat*dyhat_dz, np.transpose(x))

        # calculer le gradient de la perte par rapport à by 
        # Attention, dby doit être de dimension (n_y, 1), pensez à positionner l'attribut
        # keepdims de la fonction numpy.sum() à True !
        dby = np.sum(dy_hat*dyhat_dz, axis=1, keepdims=True)
        
        # Stocker les gradients dans un dictionnaire
        gradients = {"dx": dx, "dby": dby, "dWxy": dWxy}
        
        return gradients

    def forward(self, x_batch):
      y, cache = self.dense_layer_forward(x_batch, self.Wxy, self.by, self.activation)
      self.cache = cache
      return y

    def backward(self, dy_hat):
      return self.dense_layer_backward(dy_hat, self.Wxy, self.by, self.activation, self.cache)

    def update_parameters(self, gradients, learning_rate):
      self.Wxy = gradients['dWxy'] * learning_rate
      self.by  = gradients['dby']  * learning_rate

    def mean_square_error(y_true, y_pred):
        """
        Erreur quadratique moyenne entre prédiction et vérité-terrain

        Arguments :
        y_true -- labels à prédire (vérité-terrain), de dimension (m, n_y)
        y_pred -- prédictions du modèle, de dimension (m, n_y)
        Retourne :
        loss -- l'erreur quadratique moyenne entre y_true et y_pred, scalaire
        dy_hat -- dérivée partielle de la fonction objectif par rapport à y_pred, de dimension (m, n_y)
        """  
        loss = np.mean(np.square(y_true - y_pred))
        dy_hat = 2*(y_pred - y_true) / y_pred
        return loss, dy_hat