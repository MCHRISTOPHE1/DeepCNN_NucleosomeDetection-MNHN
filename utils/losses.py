# -*- coding: utf8 -*-

"""
	This module contains the custom losses or metrics that can be used to train or to evaluate a neural network.
	It is made to work as a usual loss or metric.
"""

try:
    from keras import backend as K
except ModuleNotFoundError:
    from tensorflow.keras import backend as K

import keras.backend as K
import tensorflow as tf
from keras.layers import Conv1D, Input, Lambda, Reshape
from keras.models import Model


def wmae(y_true, y_pred, weights):
    return K.mean(K.abs(y_true-y_pred)*weights)

def wcorrelate(y_true, y_pred, weights):
    mx = K.sum(y_true*weights)/K.sum(weights)
    my = K.sum(y_pred*weights)/K.sum(weights)
    
    sx = K.sum(weights*(y_true-mx)**2)/K.sum(weights)
    sy = K.sum(weights*(y_pred-my)**2)/K.sum(weights)
    
    sxy = K.sum(weights*(y_true-mx)*(y_pred-my))/K.sum(weights)
    return sxy/K.sqrt(sx*sy)

def wmae_cor(y_true, y_pred, weights):
    return wmae(y_true, y_pred, weights) + 1 - wcorrelate(y_true, y_pred, weights)



def correlate(y_true, y_pred):
    """
		Calculate the correlation between the predictions and the labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = correlate)
		>>> load_model('file', custom_objects = {'correlate : correlate})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    return sigma_XY/(sigma_X*sigma_Y + K.epsilon())

def max_absolute_error(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    return K.log(K.mean(K.exp(diff)))

def mae_var_cor(y_true, y_pred):
    """
	   Calculate the mean absolute error minus the correlation between
        predictions and  labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_cor)
		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
	"""

    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)

    varX = K.sum(K.square(X))/len(y_true)
    varY = K.sum(K.square(Y))/len(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))
    var = K.abs(varX - varY)
    
    return mae + 1 - cor + 1 + var

def mymse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


# def wmae_cor(y_true, y_pred):
#     """
# 	   Calculate the mean absolute error minus the correlation between
#         predictions and  labels.

# 		:Example:

# 		>>> model.compile(optimizer = 'adam', losses = mae_cor)
# 		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
# 	"""
#     X = y_true - K.mean(y_true)
#     Y = y_pred - K.mean(y_pred)
    
#     sigma_XY = K.sum(X*Y)
#     sigma_X = K.sqrt(K.sum(X*X))
#     sigma_Y = K.sqrt(K.sum(Y*Y))
    
#     cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
#     mae = K.abs(y_true - y_pred)
    
#     return ((1- cor) + mae)

def mae_cor(y_true, y_pred):
    """
	   Calculate the mean absolute error minus the correlation between
        predictions and  labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_cor)
		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))
    
    return ((1- cor) + mae)

def maex2_cor(y_true, y_pred):
    """
	   Calculate the mean absolute error minus the correlation between
        predictions and  labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_cor)
		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))
    
    return ((1- cor) + 2*mae)

def rmse_cor(y_true, y_pred):
    """
	   Calculate the mean absolute error minus the correlation between
        predictions and  labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_cor)
		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.sqrt(K.mean((y_true - y_pred)**2))
    
    return 1 + mae - cor

def rmsle(y_true, y_pred):
    X = y_true-0.5
    s = K.sign(X)
    X /= s

    Y = (y_pred-0.5)/s

    X = K.log(X+1)
    Y = K.log(Y+1)

    return K.sqrt(K.mean(X-Y))

def stretch_cor(y_true, y_pred):
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    stretch=0.5 - K.abs(Y-0.5)

    return 1-cor+stretch

def mae_cor_ratio(y_true, y_pred):
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))

    return mae/cor


def mae_cor_div(y_true, y_pred):
    """
	   Calculate the mean absolute error minus the correlation between
        predictions and  labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_cor)
		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))
    
    return mae/cor


def mae_cor_prod(y_true, y_pred):
    """
	   Calculate the mean absolute error minus the correlation between
        predictions and  labels.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_cor)
		>>> load_model('file', custom_objects = {'mae_cor : mae_cor})
	"""
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)
    
    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))
    
    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))
    
    return mae * (1 - cor)


def msle(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    alpha = 5

    log = K.mean(K.square(K.log(y_true+1e-9)-K.log(y_pred+1e-9)))

    return mse + (alpha*log)

def mse_var(y_true, y_pred) : 
    """
		Calculate the mean squared error between the predictions and the labels and add the absolute difference of
		variance between the distribution of labels and the distribution of predictions.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mse_var)
		>>> load_model('file', custom_objects = {'mse_var' : mse_var})
	"""
    X = y_true - y_pred
    
    Y = K.mean(X**2) + K.abs(K.var(y_true) - K.var(y_pred))
    
    return Y

def bray_curtis(y_true, y_pred) :
    """
		Calculate the Bray Curtis distance between the predictions and the label.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = bray_curtis)
		>>> load_model('file', custom_objects = {'bray_curtis : bray_curtis})
	"""
    X = K.sum(K.minimum(y_true, y_pred))
    
    Y = K.sum(y_true + y_pred)
    
    return (1 - 2*X/Y)

def mae_wo_zeros(y_true, y_pred) :
    """
		Calculate the mean absolute error between the predictions and the label but without taking into account the contribution
		of zeros within a sequence.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mae_wo_zeros)
		>>> load_model('file', custom_objects = {'mae_wo_zeros : mae_wo_zeros})

		..notes:: It is equivalent to set sequence weight to the sign of the sequence. A method already exist in keras but does not
				  seem to work.
	"""    
    X = y_true - y_pred
    
    sample_weight = K.sign(y_true)
    
    X_weighted = sample_weight*K.abs(X)
    
    Y = K.mean(X_weighted)
    
    return Y

def mse_wo_zeros_var(y_true, y_pred) :
    """
		Calculate the mean absolute error between the predictions and the label but without taking into account the contribution
		of zeros within a sequence. After that it adds the absolute value of the difference of variance between the prediction
		distribution and the label distribution.

		:Example:

		>>> model.compile(optimizer = 'adam', losses = mse_wo_zeros_var)
		>>> load_model('file', custom_objects = {'mse_wo_zeros_var : mse_wo_zeros_var})

		..notes:: It is equivalent to set sequence weight to the sign of the sequence and to use mse_var. A method already exists
				  in keras but does not seem to work.
	"""        
    X = y_true - y_pred
    
    sample_weight = K.sign(y_true)
    
    X_weighted = sample_weight*K.abs(X)
    
    Y = K.mean(X_weighted**2) + K.abs(K.var(y_true) - K.var(y_pred))
    
    return Y

def MCC(y_true, y_pred):
     """
    		Calculate the Mattheew correlation coefficient between the predictions and the label.
    
    		:Example:
    
    		>>> model.compile(optimizer = 'adam', losses = MCC)
    		>>> load_model('file', custom_objects = {'MCC : MCC})
    
    		..notes:: This metrics is usefull to evaluate the accuracy with imbalanced dataset.
     """        
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
    
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
     
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
    
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
    
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
     return numerator / (denominator + K.epsilon())
