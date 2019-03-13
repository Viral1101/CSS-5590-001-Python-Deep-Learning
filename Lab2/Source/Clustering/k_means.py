import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score

# Read the data in
# test = pd.read_csv('test_2v.csv')
train = pd.read_csv('train_2v.csv')

# Determine the number of missing items and drop them from the bmi column
print(train.isna().sum())
train = train.dropna(subset=['bmi'])

# Recheck the data for missing values
print(train.isna().sum())

# Split the data into x (age, bmi, avg glucose level) and y (stroke)
x = train.iloc[:,[2,8,9]]
# y = train.iloc[:-1]

# Scatter plot the data and color the points where a stroke was noted
sns.FacetGrid(train, hue='stroke', height=4).map(plt.scatter, 'age', 'bmi', 'avg_glucose_level')
plt.show()

# temp = pd.DataFrame(y['stroke'])

# Plot the data again, but in 3D space again coloring the points to denote the presence of stroke
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.age, x.bmi, x.avg_glucose_level, c=train['stroke'])
plt.show()

# Scale the x's
scaler = preprocessing.StandardScaler()
scaler.fit(x)
x_scaled_array = scaler.transform(x)
x_scaled = pd.DataFrame(x_scaled_array, columns=x.columns)

# Perform kmeans clustering on the scaled x-data
nclusters = 2
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(x_scaled)

# Predict the clusters and convert to a dataframe for plotting
y_cluster = km.predict(x_scaled)
ys = pd.DataFrame(y_cluster)

# Plot the data in 3D space and color the clusters
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.age, x.bmi, x.avg_glucose_level, c=ys[0])
plt.show()

# Elbow method to determine optimum clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.show()

# Silhouette score to evaluate the distance between clusters.
print(silhouette_score(x, y_cluster))
