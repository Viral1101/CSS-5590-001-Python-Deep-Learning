from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets

# import the iris dataset
iris = datasets.load_iris()

# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# get a reference to the Naive Bayes method and fit a model using the training data
nb = GaussianNB()
nb.fit(X_train, y_train)
# print the accuracy scores of the model when evaluated against the training and test datasets
print("Accuracy of Naive Bayes classifier on training set: {:.2f}".format(nb.score(X_train, y_train)))
print("Accuracy of Naive Bayes classifier on test set: {:.2f}".format(nb.score(X_test, y_test)))

# fit an SVM model using a linear kernel to the training data
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
# print the accuracy scores of the model when evaluated against the training and test datasets
print("Accuracy of SVM classifier on training set: {:.2f}".format(clf.score(X_train, y_train)))
print("Accuracy of SVM classifier on test set: {:.2f}".format(clf.score(X_test, y_test)))

# NB performs worse than SVM as it assumes independence between features. SVMs determine a linear
# decision boundary between classes in higher dimensions.

# fit an SVM model using a linear kernel to the training data
clf2 = SVC(kernel='rbf', C=1).fit(X_train, y_train)
# print the accuracy scores of the model when evaluated against the training and test datasets
print("Accuracy of SVM-RBF classifier on training set: {:.2f}".format(clf2.score(X_train, y_train)))
print("Accuracy of SVM-RBF classifier on test set: {:.2f}".format(clf2.score(X_test, y_test)))

# The Radial Basis Function kernel lowers the accuracy of classification in this dataset as
# the linear kernel is a parametric model and rbf is a non-parametric model.
# Parametric models generally have higher statistical power.