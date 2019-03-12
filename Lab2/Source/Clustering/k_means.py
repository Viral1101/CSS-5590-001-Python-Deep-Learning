import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score

test = pd.read_csv('test_2v.csv')
train = pd.read_csv('train_2v.csv')

print(train.isna().sum())

train = train.dropna(subset=['bmi'])

print(train.isna().sum())

x = train.iloc[:,[2,8,9]]
y = train.iloc[:-1]

sns.FacetGrid(train, hue='stroke', height=4).map(plt.scatter, 'age', 'bmi', 'avg_glucose_level')
plt.show()

temp = pd.DataFrame(y['stroke'])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.age, x.bmi, x.avg_glucose_level, c=train['stroke'])
plt.show()

scaler = preprocessing.StandardScaler()
scaler.fit(x)

x_scaled_array = scaler.transform(x)
x_scaled = pd.DataFrame(x_scaled_array, columns=x.columns)

nclusters = 3
seed = 0

km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(x_scaled)

y_cluster = km.predict(x_scaled)
ys = pd.DataFrame(y_cluster)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.age, x.bmi, x.avg_glucose_level, c=ys[0])
plt.show()

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.show()

print(silhouette_score(x, y_cluster))
