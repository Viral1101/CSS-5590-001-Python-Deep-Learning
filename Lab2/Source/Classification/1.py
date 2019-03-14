import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# Read in data set that includes numeric and non-numeric features
data = pd.read_table('adult.data', delimiter=', ')

features = data.drop('income', axis=1)
target = data['income']

# Perform exploratory data analysis on the data set
# Handle null values
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
# There are no null values included in this data set

# Encode the categorical features
data = data.apply(LabelEncoder().fit_transform)
features = features.apply(LabelEncoder().fit_transform)

# Remove features not correlated to the target class
corr = data.corr()
print(corr['income'].sort_values(ascending=False), '\n')
# The features relationship, marital-status, fnlwgt, etc. appear to have no correlation
features = features.drop(['relationship','marital-status','fnlwgt'], axis=1)

# Scale features to normalize data for faster SVM convergence
features = scale(features)

x_train, x_test, y_train, y_test = train_test_split(features, target)

# Apply the three classification algorithms
# Naïve Bayes
bayes = GaussianNB()
bayes.fit(x_train, y_train)
bayes_accuracy = bayes.score(x_test, y_test)
print("Accuracy of Naive Bayes: ", bayes_accuracy)

# SVM
svm = SVC(kernel='linear', C=1)
svm.fit(x_train, y_train)
svm_accuracy = svm.score(x_test, y_test)
print("Accuracy of Support Vector Machines: ", svm_accuracy)

# KNN
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_accuracy = knn.score(x_test, y_test)
print("Accuracy of K Nearest Neighbors: ", knn_accuracy)

# Report which classifier gives better result
# Results are obviously varied based on the random seed
# However, the KNN model tends to have a slightly higher accuracy than the others.
# All models give an accuracy of around 80% consistently.
