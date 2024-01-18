import numpy as np
import math
import matplotlib.pyplot as plt 
from DenseLayer import DenseLayer 
from sklearn.model_selection import train_test_split
from sklearn import datasets

'''
np.random.seed(1)
x_tmp = np.random.randn(3,10)
Wxy = np.random.randn(2,3)
by = np.random.randn(2,1)

activation = 'relu'
y_pred_tmp, cache_tmp = dense_layer_forward(x_tmp, Wxy, by, activation)
print("y_pred.shape = \n", y_pred_tmp.shape)

print('----------------------------')

print("activation relu : y_pred[1] =\n", y_pred_tmp[1])

print('----------------------------')

activation = 'sigmoid'
y_pred_tmp, cache_tmp = dense_layer_forward(x_tmp, Wxy, by, activation)
print("activation sigmoid : y_pred[1] =\n", y_pred_tmp[1])

print('----------------------------')

activation = 'linear'
y_pred_tmp, cache_tmp = dense_layer_forward(x_tmp, Wxy, by, activation)
print("activation linear : y_pred[1] =\n", y_pred_tmp[1])
'''
####################################################################

np.random.seed(1)
x_tmp = np.random.randn(3,10)
Wxy = np.random.randn(2,3)
by = np.random.randn(2,1)
dy_hat = np.random.randn(2, 10)
activation = 'relu'
y_pred_tmp, cache_tmp = dense_layer_forward(x_tmp, Wxy, by, activation)
gradients = dense_layer_backward(dy_hat, Wxy, by, activation, cache_tmp)
print("dimensions des différents gradients :")
print("dx : ", gradients['dx'].shape)
print("dby : ", gradients['dby'].shape)
print("dWxy : ", gradients['dWxy'].shape)

print('----------------------------')

print("activation relu : gradients =\n", gradients)

print('----------------------------')

activation = 'sigmoid'
gradients = dense_layer_backward(dy_hat, Wxy, by, activation, cache_tmp)
print("activation sigmoid : gradients =\n", gradients)

print('----------------------------')

activation = 'linear'
gradients = dense_layer_backward(dy_hat, Wxy, by, activation, cache_tmp)
print("activation linear : gradients =\n", gradients)