import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score, KFold
from sklearn import svm
import matplotlib.pyplot as plt

# Load iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a k-fold object
k = 10
kf = KFold(n_splits=k)

# Create a SVM classifier
clf = svm.SVC(kernel='linear', C=1)

# Perform k-fold cross-validation
scores = cross_val_score(clf, X, y, cv=kf)

# Visualize cross-validation scores
plt.figure()
plt.plot(range(1, k+1), scores)
plt.xlabel('Fold number')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

from sklearn.model_selection import train_test_split

# Assuming you have features X and target y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Count the number of samples in each set
n_train = len(X_train)
n_test = len(X_test)

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(['Train', 'Test'], [n_train, n_test])
plt.xlabel('Dataset')
plt.ylabel('Number of samples')
plt.title('Train-Test Split')
plt.show()

