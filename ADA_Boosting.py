import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import pickle

# Saving Stumps to File
def save_stumps(stumps):
    with open('stumps', 'wb') as file:
        pickle.dump(stumps, file)

# Reading Stumps from File
def load_stumps():
    with open('stumps', 'rb') as file:
        stumps = pickle.load(file)
    return stumps

# Creating function for calculating weighted miss-classification error (wmce)
def wmce(y_true, y_pred, weights):
    incorrect = y_true != y_pred
    error = np.sum(weights[incorrect]) / np.sum(weights)
    return error

# Creating function to calculate the amount of say (alpha)
def calc_alpha(loss):
    alpha = np.log((1 - loss) / loss)
    return alpha

# Creating function to update weights using formula w_i * e^(alpha) for all w_i that were miss_classified
def update_weights(weights, y_true, y_pred, alpha, best_split):
    incorrect =  y_true != y_pred
    weights[incorrect] = weights[incorrect]* np.exp(alpha)
    weights = weights/ np.sum(weights)
    return weights

# Creating function to compute the best split from the given data
def get_best_split(x_train_final, y_train_final, weights,x_train, y_train ,num_thresholds=300):
    
    best_split = None
    best_val = None
    best_pred = None
    min_wmce = float('inf')
    left_region_mode = None
    right_region_mode = None
    best_left_region_mode = None
    best_right_region_mode = None    

    for feature in range(5):
        feature_val = x_train_final[feature, :]
        thresholds = np.linspace(feature_val[0], feature_val[-1], num=num_thresholds)
        feature_val = x_train[feature,:]
        for val in thresholds:
            left_region = feature_val <= val
            right_region = feature_val > val
            if np.any(left_region):
                left_region_mode = mode(y_train[left_region]).mode[0]
            if np.any(right_region):
                right_region_mode = mode(y_train[right_region]).mode[0]
            y_pred = np.where(feature_val <= val, left_region_mode, right_region_mode)
            loss = wmce(y_train, y_pred, weights)
            
            if loss < min_wmce:
                min_wmce = loss
                best_split = feature
                best_val = val
                best_pred = y_pred
                best_left_region_mode = left_region_mode
                best_right_region_mode = right_region_mode
                
    return best_split, best_val, min_wmce, best_pred, best_left_region_mode, best_right_region_mode
            
# Creating decision stump
def decision_stump(x_train_final, y_train_final,weights, x_train, y_train ):
    best_split, best_val, min_wmce, y_pred, left_mode, right_mode = get_best_split(x_train_final, y_train_final, weights, x_train,y_train)
    alpha = calc_alpha(min_wmce)
    weights = update_weights(weights, y_train, y_pred, alpha, best_split)
    stump = {"split_dim": best_split, "split_val": best_val, "alpha": alpha, "left": left_mode, "right": right_mode}
    return stump, weights


#Training n number of stumps 
def create_stumps(x_train_final, y_train_final, x_train, y_train  ,num_stump):
    stumps = []  
    weights = np.ones(len(y_train_final[0])) / len(y_train_final[0])
    for i in range(num_stump):
        stump, weights = decision_stump(x_train_final, y_train_final, weights, x_train, y_train)
        stumps.append(stump) 
    save_stumps(stumps)