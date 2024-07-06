import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# Loading the dataset
data = np.load("mnist.npz")

# Check the shapes of the loaded arrays
x_train = data['x_train']
y_train = data['y_train']
x_test = data["x_test"]
y_test = data["y_test"]

print("Shape of x_train:",x_train.shape)
print("Shape of y_train:",y_train.shape)
print("Shape of x_test:",x_test.shape)
print("Shape of y_test:",y_test.shape)
# Visualize 5 samples from each class in the train set in the form of images.
fig, axes = plt.subplots(10, 5, figsize=(12, 10))
for digit in range(10):
    digit_indices = np.where(data["y_train"] == digit)[0][:5] 
    for i, index in enumerate(digit_indices):
        axes[digit, i].imshow(data["x_train"][index])
        axes[digit, i].set_title(f"Digit {digit}")
        axes[digit, i].axis('off')

plt.tight_layout()
plt.show()

# Reshape images to make them 784-dimensional
x_train_flat = data["x_train"].reshape(-1, 784)
x_test_flat = data["x_test"].reshape(-1, 784)

#Finding mean and Covariance
train_class_means = {}
train_class_cov = {}
for digit in range(10):
    x_train_class = x_train_flat[y_train == digit]
    x_train_class = x_train_class 
    expected_val = np.mean(x_train_class,axis=0)
    train_class_means[digit] = expected_val
    cov= np.cov(x_train_class, rowvar = False) + np.identity(784)*(10**-3)
    train_class_cov[digit] = cov

print("Shape of cov matrix:",train_class_cov[0].shape)
#Finding Determinant, Priors and Inverse of Covariance Matrix for computation
determinants = []
inverses = []
subset_class_priors = []
for k in range(10):
    determinants.append(np.linalg.slogdet(train_class_cov[k])[1])
    inverses.append(np.linalg.inv(train_class_cov[k]))
    subset_class_priors.append(len(x_train_flat[y_train == k]) / len(x_train_flat)) 

# Perform QDA classification
predicted_classes = []

for x in x_test_flat:
    quad_functions = []
    for k in range(10):
        # Compute quadratic discriminant function for each class
        quad_func = -0.5 * np.dot(np.dot((train_class_means[k]).T, inverses[k]), (train_class_means[k])) \
                    - 0.5 * np.log(determinants[k]) + np.log(subset_class_priors[k]) + np.dot((-0.5)*np.dot(x.T,inverses[k]),x) \
                    + np.dot(np.dot(inverses[k],train_class_means[k]).T,x)
        quad_functions.append(quad_func) 
    
    # Predict the class with maximum quadratic discriminant function
    predicted_class = np.argmax(quad_functions)
    predicted_classes.append(predicted_class)

# Compute overall accuracy
overall_accuracy = np.mean(predicted_classes == y_test)
print("Overall accuracy:", overall_accuracy)



# Compute class-wise accuracy for QDA Classification
class_accuracy = classification_report(y_test, predicted_classes, output_dict=True)
for digit in range(10):
    print(f"Class {digit} accuracy with QDA Classification:", class_accuracy[str(digit)]['precision'])


# Create and train QDA model using scikit learn for testing
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train_flat, y_train)

# Predict classes for test set
predicted_classes = qda.predict(x_test_flat)

# Compute accuracy
overall_accuracy = accuracy_score(y_test, predicted_classes)

print("Overall accuracy usign scikit learn:", overall_accuracy)

# Calculate class-wise accuracy
class_correct = {i: 0 for i in range(10)}
class_total = {i: 0 for i in range(10)}

for true_label, predicted_label in zip(y_test, predicted_classes):
    if true_label == predicted_label:
        class_correct[true_label] += 1
    class_total[true_label] += 1

# Display class-wise accuracy
for digit in range(10):
    accuracy = class_correct[digit] / class_total[digit] if class_total[digit] > 0 else 0
    print(f"Class {digit} accuracy:", accuracy)