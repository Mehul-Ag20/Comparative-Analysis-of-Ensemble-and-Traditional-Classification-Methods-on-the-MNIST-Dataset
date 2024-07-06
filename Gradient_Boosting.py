import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import pickle

# Saving Stumps to File
def save_stumps_reg(stumps):
    with open('stumps_reg', 'wb') as file:
        pickle.dump(stumps, file)

# Reading Stumps from File
def load_stumps_reg():
    with open('stumps_reg', 'rb') as file:
        stumps = pickle.load(file)
    return stumps

# Creating function for calculating sum squared residuals (ssr)
def ssr(y_left, y_right, left_mean, right_mean):
    ssr = np.sum((y_left-left_mean)**2) + np.sum((y_right-right_mean)**2)
    return ssr

#Creating function to calculate th negative gradient
def negative_gradient(y_train):
    return np.sign(y_train)

# Creating function to calculate residue after each stump ... r = r - 0.1*h1(x)
def calc_residue(y_train,y_pred):
    return y_train - 0.01 * y_pred


# Creating function to compute the best split from the given data
def get_best_split(x_train_final, y_train_final ,x_train, y_train ,num_thresholds=1000):
    
    best_split = None
    best_val = None
    min_ssr = float('inf') 

    for feature in range(5):
        feature_val = x_train_final[feature, :]
        thresholds = np.linspace(feature_val[0], feature_val[-1], num=num_thresholds)
        x_vals = x_train[feature, :]
        for val in thresholds:
            left_region = x_vals <= val
            right_region = x_vals > val
            left_mean = np.mean(y_train[left_region])
            right_mean = np.mean(y_train[right_region])
            sum_residue = ssr(y_train[left_region],y_train[right_region],left_mean,right_mean)
            
            if sum_residue < min_ssr:
                min_ssr = sum_residue
                best_split = feature
                best_val = val
                
    return best_split, best_val, min_ssr

 # Construct the decision stump
 # Note : final subscript stands for sorted data where train subscript stands for pca performed dataset
def decision_stump_regression(x_train, y_train, x_train_final, y_train_final):  #y_train here is basically residue
    y_train_gradient = negative_gradient(y_train)
    best_split, best_val, min_ssr = get_best_split(x_train_final, y_train_final,x_train,y_train_gradient)
    left_split_indices = np.where(x_train[best_split,:] <= best_val)[0]
    right_split_indices = np.where(x_train[best_split,:] > best_val)[0]
    left = np.mean(y_train_gradient[left_split_indices])
    right = np.mean(y_train_gradient[right_split_indices])
    y_pred = np.zeros(len(y_train))
    y_pred[left_split_indices] = left
    y_pred[right_split_indices] = right
    residue = calc_residue(y_train,y_pred)
    stump = {
        "split_dim": best_split,
        "split_val": best_val,
        "min_ssr": min_ssr,
        "left":left,
        "right":right
    }
    return stump, residue

#Training n number of stumps 
def create_gradien_stumps(x_train,y_train,x_train_final,y_train_final, num_stump):
    stumps = []
    residue = y_train
    for i in range(num_stump):
        stump, residue = decision_stump_regression(x_train,residue,x_train_final,y_train_final)
        stumps.append(stump)
    save_stumps_reg(stumps)
