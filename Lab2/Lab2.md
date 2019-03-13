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
### Objective
Perform a multiple regression on a new dataset and evaluate the model with R² and RMSE.
### Data
The data for this portion of the lab was obtained from the UCI machine learning repo. Here, the [air quality data set](https://archive.ics.uci.edu/ml/datasets/Air+Quality) was extracted and the CSV was edited in notepad to convert the european style decimal points to periods and the semicolons to commas to allow for easier reading by python. The headings were also adjusted into unique, easy to parse versions of the originals.
### Approach
The data was read into a panda, cleaned up a bit by dropping some columns that were not going to be used, checked for missing data, and then removing missing values.
```python
data = pd.read_csv('AirQualityUCI.csv')
data = data.drop(['delete', 'delete2', 'Date', 'Time'], axis=1)

nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

data = data.dropna()
```
The data was then split into test and training sets, and histograms were plotted to determine which variables aproximated a normal distribution; S1, NOx, and S3 were chosen. S3 appears as though it would approximate a normal distribution after a log transformation.
```python
train, test = train_test_split(data, test_size=0.33, random_state=42)

for col in data.columns:
    plt.hist(data[col], color='blue')
    plt.show()
```
![S1](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/reg_hist_S1.png)
![NOx](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/reg_hist_NOx.png)
![S3](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/reg_hist_S3.png)

This data set, evidently, uses -200 to indicate a missing value. So, the data will need to be adjusted to account for this.
```python
y = train.NOx[train.S1 > 0]
x = train.S1[train.S1 > 0]
x2 = train.S3[train.S1 > 0]

y = y[train.NOx > 0]
x = x[train.NOx > 0]
x2 = x2[train.NOx > 0]
```
We can then check the boxplot of the S1 data to remove obvious outliers (>= 2 standard deviations from the mean) and shape the remaining data based on that information.
```python
plt.boxplot(x)
plt.show()

x_avg = np.mean(x)
x_std = np.std(x)

y = y[x < 2*x_std + x_avg]
x2 = x2[x < 2*x_std + x_avg]
x = x[x < 2*x_std + x_avg]
```
![Boxplot](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/reg_boxplot.png)

The linearity of both x variables needs to be checked as well.
```python
plt.scatter(x, y, alpha=0.75, color='b')
plt.xlabel('S1')
plt.ylabel('NOx')
plt.title('NOx by S1 concentration')
plt.show()

plt.scatter(x2, y, alpha=0.75, color='b')
plt.xlabel('S3')
plt.ylabel('NOx')
plt.title('NOx by S3 concentration')
plt.show()
```
![S1 plot](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/reg_S1.png)
![S3 plot](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/reg_S3.png)

Since the S3 plot appears to have asymptotes, the data needs to be transformed to better aproximate a linear relationship. This is done here with a log10 transform.
```python
x2 = np.log10(x2)

plt.scatter(x2, y, alpha=0.75, color='b')
plt.xlabel('S3')
plt.ylabel('NOx')
plt.title('NOx by S3 concentration')
plt.show()
```
![Log10(S3)](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/reg_Log10S3.png)

The regression is done before the Log10 transform and removing outliers as well as afterwards for comparisson.
```python
x = pd.concat([x, x2], axis=1)
clf = linear_model.LinearRegression()
clf.fit(x, y)
```
The test data is prepared in a similar manner as the training data to remove the -200 values.
```python
x_test = pd.concat([test.S1, test.S3], axis=1)
y_test = test['NOx']

y_test = y_test[x_test.S1 > 0]
x_test = x_test[x_test.S1 > 0]

y_test = y_test[x_test.S3 > 0]
x_test = x_test[x_test.S3 > 0]

x_test = x_test[y_test > 0]
y_test = y_test[y_test > 0]

x_test = pd.concat([x_test.S1, np.log10(x_test.S3)], axis=1)
```
The model can then be evaluated using the fitted regression and the test data.
```python
predictions = clf.predict(x_test)
predictions2 = clf_pre.predict(x_test)

print("R² with outliers: ", clf_pre.score(x_test, y_test))
print("R²: ", clf.score(x_test, y_test))
print("RMSE with outliers: ", mean_squared_error(y_test, predictions2))
print("RMSE: ", mean_squared_error(y_test, predictions))
```
### Model Evaluation
Removing the outliers improved the skew of S1 `0.7165115084769457` to `0.3888873413151674`, and the log10 transform improved the skew of S3 from `1.3629614837490542` to `0.25495119988359766`.

The R² with outliers and before the log transform evaluated to `-0.2040498364998069`, but improved to `0.5871944116925383` after the aforementioned adjustments.
The RMSE with outliers and before the log transform evaluated to `50740.46229962097`, but improved to `17396.24536761521` after the aforementioned adjustments.
## Conclusion
