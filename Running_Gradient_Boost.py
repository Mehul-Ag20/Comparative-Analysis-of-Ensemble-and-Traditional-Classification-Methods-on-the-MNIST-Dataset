import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import pickle
from Gradient_Boosting import create_gradien_stumps, load_stumps_reg

# Loading the dataset
data = np.load("mnist.npz")

# Check the shapes of the loaded arrays
x_train = data['x_train']
y_train = data['y_train']
x_test = data["x_test"]
y_test = data["y_test"]

#Reshaping x_train to 784,60000
x_train_flat =x_train.reshape(-1, 784)

# Selecting classes 0,1  from the original data
x_train_new = []
y_train_new = []
val_x = []
val_y= []
for digit in range(2):
    select_digit = np.where(y_train==digit)[0]
    get_index =  select_digit[:1000]
    for i in select_digit:
        if (i not in get_index):
            x_train_new.append(x_train_flat[i])
            y_train_new.append(y_train[i])
        else:
            val_x.append(x_train_flat[i])
            val_y.append(y_train[i])

x_train_new = np.array(x_train_new)
y_train_new = np.array(y_train_new)

#Creating validation set of 2000 sample datapoints from training daatset
val_x = np.array(val_x)
val_y = np.array(val_y)

#Computing covariance ,eigenvectors and eigenvalue. Applying PCA on the centralized PCA.
x_train_centered = x_train_new - np.mean(x_train_new, axis=0)
num_of_samples = 10665
cov_matrix = np.dot(x_train_centered.T, x_train_centered) / (num_of_samples -1)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sort_ind = np.argsort(eigenvalues)[::-1] #Sorting in descending order
sorted_eigenvalues = eigenvalues[sort_ind]
U = eigenvectors[:, sort_ind]

#Applying PCA to reduce the dimension to p=5
Up = U[:, :5]
Yp = np.dot(Up.T, x_train_centered.T)
Xrecon_p = np.dot(Up, Yp).T + np.mean(x_train_new, axis=0)
Xrecon_p = Xrecon_p.T
x_train_pca = Yp

#Sorting the data in ascending order
x_train_final = np.zeros((5, 10665))
y_train_final = np.zeros((5,10665))

for dim in range(5):
    sorted_indices = np.argsort(Yp[dim])
    x_train_final[dim] = Yp[dim][sorted_indices]
    y_train_final[dim] = y_train_new[sorted_indices]

#Removing mean from y-train to obtain the negative gradient 
y_train_gradient = y_train_new - np.mean(y_train_new)

x_train_pca = Yp
num_stumps = 300
# create_gradien_stumps(x_train_pca,y_train_gradient,x_train_final,y_train_final,num_stumps)
stumps_reg = load_stumps_reg()

# Evaluate MSE on the validation set after each iteration
mse_list = []
pred = np.ones(len(val_y))* np.mean(val_y)
val_x_centered = val_x - np.mean(val_x,axis=0)
val_x_new = np.dot(Up.T, val_x_centered.T)
for stump in stumps_reg:
    split_dim, split_val, left, right = stump["split_dim"], stump["split_val"], stump["left"], stump["right"]
    y_pred = np.where(val_x_new[split_dim] <= split_val, left, right)
    pred += 0.01*y_pred
    mse = np.mean((val_y-pred)**2)
    mse_list.append(mse)

# Plot mse on validation set vs. number of trees
plt.plot(range(1, num_stumps+ 1), mse_list, marker='o', linestyle='-')
plt.xlabel('Number of Trees')
plt.ylabel('Validation MSE')
plt.title('Validation MSE Set vs. Number of Trees')
plt.grid(True)
plt.show()

#get the num_stump with minimum mse 
min_mse_index = np.argmin(mse_list)
num_stumps_to_use = min_mse_index + 1
selected_stumps = stumps_reg[:num_stumps_to_use]
print("Number of stumps to use based on maximum accuracy:", num_stumps_to_use)

# Apply PCA to reduce dimensionality of test set
x_test_flat = x_test.reshape(-1,784)
x_test_centered = x_test_flat - np.mean(x_test_flat,axis=0)
class_mask_test = np.isin(y_test, [0, 1])
x_test_new = x_test_centered[class_mask_test]
y_test_new = y_test[class_mask_test]

cov_matrix_test = np.dot(x_test_centered.T, x_test_centered) / (x_test_new.shape[0]-1)
eigenvalues_test, eigenvectors_test = np.linalg.eig(cov_matrix_test)
sort_ind_test = np.argsort(eigenvalues_test)[::-1] #Sorting in descending order
sorted_eigenvalues = eigenvalues_test[sort_ind_test]
U_test = eigenvectors[:, sort_ind_test]

Up_test = U_test[:, :5]
Yp_test = np.dot(Up_test.T, x_test_new.T)
Xrecon_p_test = np.dot(Up_test, Yp_test)

#Testing set - MSE calculation
test_mse_list = []
pred = np.ones(len(y_test_new))* np.mean(y_test_new)
for stump in selected_stumps:
    split_dim, split_val, left, right = stump["split_dim"], stump["split_val"], stump["left"], stump["right"]
    y_pred = np.where(Yp_test[split_dim] <= split_val, left, right)
    pred += 0.01*y_pred
    mse = np.mean((y_test_new-pred)**2)
    test_mse_list.append(mse)
print('TEST MSE:', np.min(test_mse_list))

# Plot mse on test dataset vs. number of stumps from slected stumps that return min mse
plt.plot(range(1, num_stumps_to_use+ 1), test_mse_list, marker='o', linestyle='-')
plt.xlabel('Number of Trees')
plt.ylabel('Test MSE')
plt.title('Test MSE vs. Number of Trees')
plt.grid(True)
plt.show()