---
layout: single
title:  "Building Random Forest Model"
categories: Python
tag: ["Random Forest Model", 'GridSearchCV', 'Visualization', 'Bootstrapping','Bagging']
toc: true
toc_sticky: true
toc_label: "Table of Contents"
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }
    
    table.dataframe td {
      text-align: center;
      padding: 8px;
    }
    
    table.dataframe tr:hover {
      background: #b8d1f3; 
    }
    
    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


### Introduction


This post will discuss building a **Random Forest Model**. Unlike *the voting* method from a different post, **Random Forest Model** is based on the *bagging* system. *Bagging* Ensemble model uses a same estimator. However, the *bagging* method uses different, or subset with duplication, of datasets when training the estimator. This is called *Boostrapping*. *Bagging* is a short-term for bootstrap aggregating. Again, I will use the same breast cancer dataset from [Kaggle](https://www.kaggle.com/datasets/sarahvch/breast-cancer-wisconsin-prognostic-data-set). 


##### Random Forest Model


**Random Forest Model** is the most popular bagging algorithm. It is based on a decision tree model, and contains the positive features of decision tree model. Also by forming several decision tree models, **Random Forest Model** can eliminate or overcome the negative sides of decision tree model itself. But, the negative side of **Random Forest Model** is that it has various parameters to set to build a model.


### Loading Dataset



```python
import pandas as pd
import numpy as np

cancer = pd.read_csv("/Users/cheolmin/Documents/machine_learning/Blog_Post/Ensemble_Voting/breast_cancer_wisconsin.csv")
cancer
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.30010</td>
      <td>0.14710</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.16220</td>
      <td>0.66560</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.08690</td>
      <td>0.07017</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.12380</td>
      <td>0.18660</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.19740</td>
      <td>0.12790</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.14440</td>
      <td>0.42450</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.24140</td>
      <td>0.10520</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.20980</td>
      <td>0.86630</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.19800</td>
      <td>0.10430</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.13740</td>
      <td>0.20500</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>564</th>
      <td>926424</td>
      <td>M</td>
      <td>21.56</td>
      <td>22.39</td>
      <td>142.00</td>
      <td>1479.0</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.24390</td>
      <td>0.13890</td>
      <td>...</td>
      <td>26.40</td>
      <td>166.10</td>
      <td>2027.0</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.4107</td>
      <td>0.2216</td>
      <td>0.2060</td>
      <td>0.07115</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>565</th>
      <td>926682</td>
      <td>M</td>
      <td>20.13</td>
      <td>28.25</td>
      <td>131.20</td>
      <td>1261.0</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.14400</td>
      <td>0.09791</td>
      <td>...</td>
      <td>38.25</td>
      <td>155.00</td>
      <td>1731.0</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.3215</td>
      <td>0.1628</td>
      <td>0.2572</td>
      <td>0.06637</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>566</th>
      <td>926954</td>
      <td>M</td>
      <td>16.60</td>
      <td>28.08</td>
      <td>108.30</td>
      <td>858.1</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.09251</td>
      <td>0.05302</td>
      <td>...</td>
      <td>34.12</td>
      <td>126.70</td>
      <td>1124.0</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.3403</td>
      <td>0.1418</td>
      <td>0.2218</td>
      <td>0.07820</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>567</th>
      <td>927241</td>
      <td>M</td>
      <td>20.60</td>
      <td>29.33</td>
      <td>140.10</td>
      <td>1265.0</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.35140</td>
      <td>0.15200</td>
      <td>...</td>
      <td>39.42</td>
      <td>184.60</td>
      <td>1821.0</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.9387</td>
      <td>0.2650</td>
      <td>0.4087</td>
      <td>0.12400</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>568</th>
      <td>92751</td>
      <td>B</td>
      <td>7.76</td>
      <td>24.54</td>
      <td>47.92</td>
      <td>181.0</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>...</td>
      <td>30.37</td>
      <td>59.16</td>
      <td>268.6</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2871</td>
      <td>0.07039</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 33 columns</p>
</div>



```python
cancer.drop(columns = ['id'],inplace = True)
cancer
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.30010</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.16220</td>
      <td>0.66560</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.08690</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.12380</td>
      <td>0.18660</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.19740</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.14440</td>
      <td>0.42450</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.24140</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.20980</td>
      <td>0.86630</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.19800</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.13740</td>
      <td>0.20500</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>564</th>
      <td>M</td>
      <td>21.56</td>
      <td>22.39</td>
      <td>142.00</td>
      <td>1479.0</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.24390</td>
      <td>0.13890</td>
      <td>0.1726</td>
      <td>...</td>
      <td>26.40</td>
      <td>166.10</td>
      <td>2027.0</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.4107</td>
      <td>0.2216</td>
      <td>0.2060</td>
      <td>0.07115</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>565</th>
      <td>M</td>
      <td>20.13</td>
      <td>28.25</td>
      <td>131.20</td>
      <td>1261.0</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.14400</td>
      <td>0.09791</td>
      <td>0.1752</td>
      <td>...</td>
      <td>38.25</td>
      <td>155.00</td>
      <td>1731.0</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.3215</td>
      <td>0.1628</td>
      <td>0.2572</td>
      <td>0.06637</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>566</th>
      <td>M</td>
      <td>16.60</td>
      <td>28.08</td>
      <td>108.30</td>
      <td>858.1</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.09251</td>
      <td>0.05302</td>
      <td>0.1590</td>
      <td>...</td>
      <td>34.12</td>
      <td>126.70</td>
      <td>1124.0</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.3403</td>
      <td>0.1418</td>
      <td>0.2218</td>
      <td>0.07820</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>567</th>
      <td>M</td>
      <td>20.60</td>
      <td>29.33</td>
      <td>140.10</td>
      <td>1265.0</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.35140</td>
      <td>0.15200</td>
      <td>0.2397</td>
      <td>...</td>
      <td>39.42</td>
      <td>184.60</td>
      <td>1821.0</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.9387</td>
      <td>0.2650</td>
      <td>0.4087</td>
      <td>0.12400</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>568</th>
      <td>B</td>
      <td>7.76</td>
      <td>24.54</td>
      <td>47.92</td>
      <td>181.0</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.1587</td>
      <td>...</td>
      <td>30.37</td>
      <td>59.16</td>
      <td>268.6</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2871</td>
      <td>0.07039</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 32 columns</p>
</div>


### Data Preprocessing



```python
cancer = cancer.iloc[:,:-1]
```


```python
cancer_target = cancer['diagnosis']
cancer_target.replace({'M':0, "B":1}, inplace = True)
cancer.drop(columns = ['diagnosis'],inplace = True)
```


```python
from sklearn.model_selection import train_test_split

X_tn, X_te, y_tn, y_te = train_test_split(cancer, cancer_target, test_size = 0.2, random_state = 11)
```


```python
y_tn.astype('category')
```

<pre>
2      0
74     1
456    1
564    0
411    1
      ..
332    1
269    1
337    0
91     0
80     1
Name: diagnosis, Length: 455, dtype: category
Categories (2, int64): [0, 1]
</pre>
### Building Random Forest Model


RandomForestClassifier(n_estimators = , max_features = ,)

- n_estimator: # of decision tree in the model. default is 10. More decision tree model tends to have better result but it is not always true. Also, running time will be greater

- max_features: # of features to decide when dividing the tree. Unlike decision tree model, default value is 'auto'.

- max_depth, min_samples_leaf, min_samples_split are same as decision tree model (to prevent overfitting).

- n_jobs = -1: This will allow to utilize all CPU Core to do machine learning



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_clf = RandomForestClassifier(n_estimators = 3,random_state = 0)
rf_clf.fit(X_tn, y_tn)
```

<pre>
RandomForestClassifier(n_estimators=3, random_state=0)
</pre>
### Evaluation



```python
y_pred = rf_clf.predict(X_te)
accuracy = accuracy_score(y_te, y_pred)
print(f"Accuracy of Random Forest Model is {np.round(accuracy,2)*100}%.")
```

<pre>
Accuracy of Random Forest Model is 95.0%.
</pre>
### Building Random Forest Model with GridSearchCV



```python
from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [8,10,12,16],
    'min_samples_leaf': [1,3,6,8,12],
    'min_samples_split': [2,4,6,8]
}
new_rf = RandomForestClassifier(n_estimators = 40, random_state = 0, n_jobs = -1)
rf_gsCV = GridSearchCV(new_rf, param_grid = params, cv = 2, n_jobs = -1)
rf_gsCV.fit(X_tn,y_tn)
```

<pre>
GridSearchCV(cv=2,
             estimator=RandomForestClassifier(n_estimators=40, n_jobs=-1,
                                              random_state=0),
             n_jobs=-1,
             param_grid={'max_depth': [8, 10, 12, 16],
                         'min_samples_leaf': [1, 3, 6, 8, 12],
                         'min_samples_split': [2, 4, 6, 8]})
</pre>

```python
print(f"Best Parameter: {rf_gsCV.best_params_}")
print(f"Best Accuracy: {rf_gsCV.best_score_}")
```

<pre>
Best Parameter: {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best Accuracy: 0.9560244222892031
</pre>

```python
y_newpred = rf_gsCV.predict(X_te)
print(f"New Accuracy with GridSearchCV: {accuracy_score(y_te, y_newpred)}")
```

<pre>
New Accuracy with GridSearchCV: 1.0
</pre>
### Visualizing Features in Importance



```python
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
rf = RandomForestClassifier(n_estimators = 40, min_samples_leaf = 1, max_depth = 8, min_samples_split = 2, random_state = 0)
rf.fit(X_tn,y_tn)
feature = rf.feature_importances_
ftr = pd.Series(feature, X_tn.columns)
ftr_top = ftr.sort_values(ascending = False).iloc[:10]
plt.figure(figsize = (10,10))
plt.title("Top 10 Feature Importances")
sb.barplot(x = ftr_top, y = ftr_top.index)
plt.show()
```

<img src="data:image/png;base64,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"/>

### Conclusion


Aggregating various decision tree models, **Random Forest Model** allows to keep the decision tree's strength and minimize the weakness as well. Using **Random Forest Model** can be a great option to use as an initial start. Also using CPU can expedite the running process. 

