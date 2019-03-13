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
![Stroke is yellow](https://github.com/Viral1101/CSS-5590-001-Python-Deep-Learning/blob/master/Lab2/Documentation/kmeans_3D.png)

Since the data points positive for stroke appear to exist in a cluster, it may be useful to cluster on these traits to identify a risk category.
## NLP
## Regression
