# Lab 2
## Introduction
The second lab assignment makes use of various machine learning techniques such as classification, clustering, natural language processing, and regression on different datasets. These will be addressed using the SciKitLearn, MatPlotLib, and Seaborn libraries.

Josh Reeves prepared the classification and NLP sections. Clustering and regression were performed by Dave Walsh.
## Classification
## Clustering
### Objective
Using an obtained dataset, perform k-means clustering, visualize the data, and demonstrate why a particular K was chosen.
### Data
The data for this portion was obtained from Kaggle and only the training data set is used. [HealthCare Problem: Prediction Stroke Patients](https://www.kaggle.com/asaumya/healthcare-problem-prediction-stroke-patients/data)
### Approach
First, the data is read in to a panda and missing data is identified and dropped. 
```python
train = pd.read_csv('train_2v.csv')
print(train.isna().sum())
train = train.dropna(subset=['bmi'])
```
We're going to look at 3 dimensions, so the independent variables are selected from the dataset and plotted. Here we're using age, bmi, and average glucose level. These data points will then be colored by the presence of stroke.
```python
x = train.iloc[:,[2,8,9]]
sns.FacetGrid(train, hue='stroke', height=4).map(plt.scatter, 'age', 'bmi', 'avg_glucose_level')
plt.show()
```
![Stroke is orange](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/kmeans_2D.png)

We can similarly plot in 3-dimensions. Here, stroke is indicated by yellow.
```python
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.age, x.bmi, x.avg_glucose_level, c=train['stroke'])
plt.show()
```
![Stroke is yellow](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/kmeans_3D.png)

Since the data points positive for stroke appear to exist in a cluster, it may be useful to cluster on these traits to identify a risk category. To accomplish this, the x-variables need to be scaled, a k is chosen, a seed is set for repeatability, and the model is fit.

```python
scaler = preprocessing.StandardScaler()
scaler.fit(x)
x_scaled_array = scaler.transform(x)
x_scaled = pd.DataFrame(x_scaled_array, columns=x.columns)

nclusters = 2
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(x_scaled)
```
Next, the clusters are predicted from the data and converted to a dataframe to plot.
```python
y_cluster = km.predict(x_scaled)
ys = pd.DataFrame(y_cluster)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x.age, x.bmi, x.avg_glucose_level, c=ys[0])
plt.show()
```
![k = 2](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/kmeans_clusters_k2_3D.png)

Finally, the chosen K is evaluated using an elbow plot and silhouette score.
```python
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.show()

print(silhouette_score(x, y_cluster))
```
![Elbow plot](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/kmeans_elbow.png)
### Model evaluation
The elbow plot indicates that the ideal K would be 2. The silhouette score for a k of 2 is `0.19910761377088917`, which indicates there isn't much distinction between the clusters; the plot confirms this score.

A K of 3 may be more justified, the benefits, according to the elbow plot, aren't as great as 2. Modeling in this manner results in a slightly more convoluted plot, and a silhouette score of `0.3385716824438738`. Since this isn't much of an improvement, and 2 clusters is more intuitive, the correct k is 2.
![K=3](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/kmeans_clusters_3D.png)
## NLP
## Regression
