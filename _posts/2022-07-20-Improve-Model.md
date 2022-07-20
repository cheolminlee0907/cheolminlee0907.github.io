---
layout: single
title:  "How to Improve Model"
categories: 
  - "Deep Learning"
tag: ["Python",'Early Stopping', 'Over Fit', "Validation Set"]
toc: true
toc_sticky: true
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


## Introduction


This post contains how to utilize **validation set** to improve a quality of machine learning model. Here, I used wine quality data from [Kaggle](https://www.kaggle.com/code/kendravanhardenberg/wine-quality-prediction).


## Loading Dataset and packages



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
wine = pd.read_csv('/Users/cheolmin/Documents/machine_learning/Blog_Post/Wine_Quality_Test/WineQT.csv')
wine
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>Id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.700</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.880</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.99680</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.760</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.99700</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.280</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.99800</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.700</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>4</td>
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
    </tr>
    <tr>
      <th>1138</th>
      <td>6.3</td>
      <td>0.510</td>
      <td>0.13</td>
      <td>2.3</td>
      <td>0.076</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>0.99574</td>
      <td>3.42</td>
      <td>0.75</td>
      <td>11.0</td>
      <td>6</td>
      <td>1592</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>6.8</td>
      <td>0.620</td>
      <td>0.08</td>
      <td>1.9</td>
      <td>0.068</td>
      <td>28.0</td>
      <td>38.0</td>
      <td>0.99651</td>
      <td>3.42</td>
      <td>0.82</td>
      <td>9.5</td>
      <td>6</td>
      <td>1593</td>
    </tr>
    <tr>
      <th>1140</th>
      <td>6.2</td>
      <td>0.600</td>
      <td>0.08</td>
      <td>2.0</td>
      <td>0.090</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99490</td>
      <td>3.45</td>
      <td>0.58</td>
      <td>10.5</td>
      <td>5</td>
      <td>1594</td>
    </tr>
    <tr>
      <th>1141</th>
      <td>5.9</td>
      <td>0.550</td>
      <td>0.10</td>
      <td>2.2</td>
      <td>0.062</td>
      <td>39.0</td>
      <td>51.0</td>
      <td>0.99512</td>
      <td>3.52</td>
      <td>0.76</td>
      <td>11.2</td>
      <td>6</td>
      <td>1595</td>
    </tr>
    <tr>
      <th>1142</th>
      <td>5.9</td>
      <td>0.645</td>
      <td>0.12</td>
      <td>2.0</td>
      <td>0.075</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99547</td>
      <td>3.57</td>
      <td>0.71</td>
      <td>10.2</td>
      <td>5</td>
      <td>1597</td>
    </tr>
  </tbody>
</table>
<p>1143 rows × 13 columns</p>
</div>


## Data Wrangling



```python
wine.drop(columns = 'Id',inplace = True)
```


```python
wine['quality'].value_counts()
```

<pre>
5    483
6    462
7    143
4     33
8     16
3      6
Name: quality, dtype: int64
</pre>
I dropped unnecessary ID columns and our target column, quality, is divided into 6 categories based on 11 features.



```python
wine.isnull().sum()
```

<pre>
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64
</pre>

```python
wine.describe()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
      <td>1143.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.311111</td>
      <td>0.531339</td>
      <td>0.268364</td>
      <td>2.532152</td>
      <td>0.086933</td>
      <td>15.615486</td>
      <td>45.914698</td>
      <td>0.996730</td>
      <td>3.311015</td>
      <td>0.657708</td>
      <td>10.442111</td>
      <td>5.657043</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.747595</td>
      <td>0.179633</td>
      <td>0.196686</td>
      <td>1.355917</td>
      <td>0.047267</td>
      <td>10.250486</td>
      <td>32.782130</td>
      <td>0.001925</td>
      <td>0.156664</td>
      <td>0.170399</td>
      <td>1.082196</td>
      <td>0.805824</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.392500</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>21.000000</td>
      <td>0.995570</td>
      <td>3.205000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.250000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>13.000000</td>
      <td>37.000000</td>
      <td>0.996680</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.100000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>61.000000</td>
      <td>0.997845</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>68.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>


I don't see any unrealistic data in this chart, and there isn't any null value that needs to be filled/adjusted.



```python
X = wine.iloc[:,:-1]
y = wine.iloc[:,-1]
y = y.astype('category')
```


```python
y.value_counts()
y = y.astype(dtype = 'category')
y
```

<pre>
0       5
1       5
2       5
3       6
4       5
       ..
1138    6
1139    6
1140    5
1141    6
1142    5
Name: quality, Length: 1143, dtype: category
Categories (6, int64): [3, 4, 5, 6, 7, 8]
</pre>
## Validation Set


You can add validation set by adding **validation_split** in **model.fit()** function. We will make 80% of training set, and set 25% of training set as validation set.



```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```


```python
X_tn, X_te, y_tn, y_te = train_test_split(X,y, test_size = 0.2, shuffle = True)
```


```python
X_tn
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>638</th>
      <td>7.4</td>
      <td>0.635</td>
      <td>0.10</td>
      <td>2.4</td>
      <td>0.080</td>
      <td>16.0</td>
      <td>33.0</td>
      <td>0.99736</td>
      <td>3.58</td>
      <td>0.69</td>
      <td>10.8</td>
    </tr>
    <tr>
      <th>476</th>
      <td>8.2</td>
      <td>0.730</td>
      <td>0.21</td>
      <td>1.7</td>
      <td>0.074</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>0.99680</td>
      <td>3.20</td>
      <td>0.52</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>473</th>
      <td>8.3</td>
      <td>0.490</td>
      <td>0.36</td>
      <td>1.8</td>
      <td>0.222</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>0.99800</td>
      <td>3.18</td>
      <td>0.60</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>1014</th>
      <td>7.0</td>
      <td>0.540</td>
      <td>0.00</td>
      <td>2.1</td>
      <td>0.079</td>
      <td>39.0</td>
      <td>55.0</td>
      <td>0.99560</td>
      <td>3.39</td>
      <td>0.84</td>
      <td>11.4</td>
    </tr>
    <tr>
      <th>326</th>
      <td>8.3</td>
      <td>0.615</td>
      <td>0.22</td>
      <td>2.6</td>
      <td>0.087</td>
      <td>6.0</td>
      <td>19.0</td>
      <td>0.99820</td>
      <td>3.26</td>
      <td>0.61</td>
      <td>9.3</td>
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
    </tr>
    <tr>
      <th>120</th>
      <td>7.9</td>
      <td>0.885</td>
      <td>0.03</td>
      <td>1.8</td>
      <td>0.058</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>0.99720</td>
      <td>3.36</td>
      <td>0.33</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>8.8</td>
      <td>0.610</td>
      <td>0.30</td>
      <td>2.8</td>
      <td>0.088</td>
      <td>17.0</td>
      <td>46.0</td>
      <td>0.99760</td>
      <td>3.26</td>
      <td>0.51</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>791</th>
      <td>7.1</td>
      <td>0.390</td>
      <td>0.12</td>
      <td>2.1</td>
      <td>0.065</td>
      <td>14.0</td>
      <td>24.0</td>
      <td>0.99252</td>
      <td>3.30</td>
      <td>0.53</td>
      <td>13.3</td>
    </tr>
    <tr>
      <th>725</th>
      <td>8.1</td>
      <td>0.820</td>
      <td>0.00</td>
      <td>4.1</td>
      <td>0.095</td>
      <td>5.0</td>
      <td>14.0</td>
      <td>0.99854</td>
      <td>3.36</td>
      <td>0.53</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>898</th>
      <td>5.0</td>
      <td>0.380</td>
      <td>0.01</td>
      <td>1.6</td>
      <td>0.048</td>
      <td>26.0</td>
      <td>60.0</td>
      <td>0.99084</td>
      <td>3.70</td>
      <td>0.75</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
<p>914 rows × 11 columns</p>
</div>



```python
model = Sequential([
    Dense(30, input_dim = 11, activation = 'relu'),
    Dense(20, activation = 'relu'),
    Dense(15, activation = 'relu'),
    Dense(10, activation = 'softmax')
])
model.summary()
```

<pre>
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_8 (Dense)             (None, 30)                360       

 dense_9 (Dense)             (None, 20)                620       
                                                                 
 dense_10 (Dense)            (None, 15)                315       
                                                                 
 dense_11 (Dense)            (None, 10)                160       
                                                                 
=================================================================
Total params: 1,455
Trainable params: 1,455
Non-trainable params: 0
_________________________________________________________________
</pre>
### Unexpected Error



```python
model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'categorical_crossentropy')
model.fit(X_tn, y_tn, epochs =50, batch_size = 10)
```

<pre>
Epoch 1/50
</pre>
Here, I was getting  **ValueError: Shapes (None, 1) and (None, 10) are incompatible.**

I am not sure why but I solved this by changing the metrics.



```python
import tensorflow
```


```python
model.compile(optimizer=tensorflow.keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy', 
              metrics=[tensorflow.keras.metrics.SparseCategoricalAccuracy()])
```


```python
model.fit(X_tn,y_tn, validation_split = 0.25, epochs = 50, batch_size = 500)
```

<pre>
Epoch 1/50
2/2 [==============================] - 0s 124ms/step - loss: 6.4811 - sparse_categorical_accuracy: 0.0161 - val_loss: 6.0581 - val_sparse_categorical_accuracy: 0.0218
Epoch 2/50
2/2 [==============================] - 0s 20ms/step - loss: 5.5498 - sparse_categorical_accuracy: 0.0146 - val_loss: 5.2768 - val_sparse_categorical_accuracy: 0.0175
Epoch 3/50
2/2 [==============================] - 0s 15ms/step - loss: 4.7714 - sparse_categorical_accuracy: 0.0146 - val_loss: 4.6969 - val_sparse_categorical_accuracy: 0.0175
Epoch 4/50
2/2 [==============================] - 0s 16ms/step - loss: 4.2064 - sparse_categorical_accuracy: 0.0482 - val_loss: 4.2431 - val_sparse_categorical_accuracy: 0.1092
Epoch 5/50
2/2 [==============================] - 0s 16ms/step - loss: 3.7751 - sparse_categorical_accuracy: 0.1416 - val_loss: 3.8749 - val_sparse_categorical_accuracy: 0.1790
Epoch 6/50
2/2 [==============================] - 0s 17ms/step - loss: 3.4396 - sparse_categorical_accuracy: 0.2307 - val_loss: 3.5427 - val_sparse_categorical_accuracy: 0.1878
Epoch 7/50
2/2 [==============================] - 0s 16ms/step - loss: 3.1394 - sparse_categorical_accuracy: 0.2467 - val_loss: 3.2301 - val_sparse_categorical_accuracy: 0.2052
Epoch 8/50
2/2 [==============================] - 0s 15ms/step - loss: 2.8667 - sparse_categorical_accuracy: 0.2584 - val_loss: 2.9200 - val_sparse_categorical_accuracy: 0.2052
Epoch 9/50
2/2 [==============================] - 0s 18ms/step - loss: 2.6087 - sparse_categorical_accuracy: 0.2759 - val_loss: 2.6137 - val_sparse_categorical_accuracy: 0.2140
Epoch 10/50
2/2 [==============================] - 0s 15ms/step - loss: 2.3525 - sparse_categorical_accuracy: 0.2803 - val_loss: 2.3205 - val_sparse_categorical_accuracy: 0.2183
Epoch 11/50
2/2 [==============================] - 0s 15ms/step - loss: 2.1066 - sparse_categorical_accuracy: 0.2905 - val_loss: 2.0456 - val_sparse_categorical_accuracy: 0.2314
Epoch 12/50
2/2 [==============================] - 0s 16ms/step - loss: 1.8910 - sparse_categorical_accuracy: 0.3226 - val_loss: 1.7940 - val_sparse_categorical_accuracy: 0.3624
Epoch 13/50
2/2 [==============================] - 0s 17ms/step - loss: 1.6913 - sparse_categorical_accuracy: 0.3942 - val_loss: 1.5855 - val_sparse_categorical_accuracy: 0.3712
Epoch 14/50
2/2 [==============================] - 0s 17ms/step - loss: 1.5236 - sparse_categorical_accuracy: 0.4117 - val_loss: 1.4293 - val_sparse_categorical_accuracy: 0.4760
Epoch 15/50
2/2 [==============================] - 0s 17ms/step - loss: 1.4099 - sparse_categorical_accuracy: 0.4701 - val_loss: 1.3355 - val_sparse_categorical_accuracy: 0.5197
Epoch 16/50
2/2 [==============================] - 0s 16ms/step - loss: 1.3446 - sparse_categorical_accuracy: 0.4701 - val_loss: 1.2900 - val_sparse_categorical_accuracy: 0.4978
Epoch 17/50
2/2 [==============================] - 0s 17ms/step - loss: 1.3078 - sparse_categorical_accuracy: 0.4555 - val_loss: 1.2639 - val_sparse_categorical_accuracy: 0.5066
Epoch 18/50
2/2 [==============================] - 0s 17ms/step - loss: 1.2714 - sparse_categorical_accuracy: 0.4482 - val_loss: 1.2378 - val_sparse_categorical_accuracy: 0.5109
Epoch 19/50
2/2 [==============================] - 0s 16ms/step - loss: 1.2264 - sparse_categorical_accuracy: 0.4672 - val_loss: 1.2228 - val_sparse_categorical_accuracy: 0.5459
Epoch 20/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1853 - sparse_categorical_accuracy: 0.4891 - val_loss: 1.2283 - val_sparse_categorical_accuracy: 0.5502
Epoch 21/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1693 - sparse_categorical_accuracy: 0.4978 - val_loss: 1.2480 - val_sparse_categorical_accuracy: 0.5066
Epoch 22/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1695 - sparse_categorical_accuracy: 0.5109 - val_loss: 1.2684 - val_sparse_categorical_accuracy: 0.4891
Epoch 23/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1742 - sparse_categorical_accuracy: 0.4993 - val_loss: 1.2750 - val_sparse_categorical_accuracy: 0.4672
Epoch 24/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1701 - sparse_categorical_accuracy: 0.5007 - val_loss: 1.2655 - val_sparse_categorical_accuracy: 0.4978
Epoch 25/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1602 - sparse_categorical_accuracy: 0.4905 - val_loss: 1.2504 - val_sparse_categorical_accuracy: 0.5284
Epoch 26/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1495 - sparse_categorical_accuracy: 0.5066 - val_loss: 1.2405 - val_sparse_categorical_accuracy: 0.5415
Epoch 27/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1444 - sparse_categorical_accuracy: 0.5080 - val_loss: 1.2376 - val_sparse_categorical_accuracy: 0.5284
Epoch 28/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1460 - sparse_categorical_accuracy: 0.4949 - val_loss: 1.2376 - val_sparse_categorical_accuracy: 0.5328
Epoch 29/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1441 - sparse_categorical_accuracy: 0.4920 - val_loss: 1.2361 - val_sparse_categorical_accuracy: 0.5197
Epoch 30/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1378 - sparse_categorical_accuracy: 0.4978 - val_loss: 1.2362 - val_sparse_categorical_accuracy: 0.4978
Epoch 31/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1327 - sparse_categorical_accuracy: 0.5182 - val_loss: 1.2404 - val_sparse_categorical_accuracy: 0.4760
Epoch 32/50
2/2 [==============================] - 0s 18ms/step - loss: 1.1311 - sparse_categorical_accuracy: 0.5124 - val_loss: 1.2433 - val_sparse_categorical_accuracy: 0.4454
Epoch 33/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1317 - sparse_categorical_accuracy: 0.5066 - val_loss: 1.2403 - val_sparse_categorical_accuracy: 0.4498
Epoch 34/50
2/2 [==============================] - 0s 17ms/step - loss: 1.1295 - sparse_categorical_accuracy: 0.5109 - val_loss: 1.2328 - val_sparse_categorical_accuracy: 0.4803
Epoch 35/50
2/2 [==============================] - 0s 18ms/step - loss: 1.1264 - sparse_categorical_accuracy: 0.5139 - val_loss: 1.2253 - val_sparse_categorical_accuracy: 0.4978
Epoch 36/50
2/2 [==============================] - 0s 18ms/step - loss: 1.1227 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.2198 - val_sparse_categorical_accuracy: 0.5022
Epoch 37/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1210 - sparse_categorical_accuracy: 0.5022 - val_loss: 1.2158 - val_sparse_categorical_accuracy: 0.5197
Epoch 38/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1198 - sparse_categorical_accuracy: 0.5124 - val_loss: 1.2136 - val_sparse_categorical_accuracy: 0.5284
Epoch 39/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1190 - sparse_categorical_accuracy: 0.5139 - val_loss: 1.2127 - val_sparse_categorical_accuracy: 0.5240
Epoch 40/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1177 - sparse_categorical_accuracy: 0.5095 - val_loss: 1.2124 - val_sparse_categorical_accuracy: 0.5240
Epoch 41/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1164 - sparse_categorical_accuracy: 0.5109 - val_loss: 1.2128 - val_sparse_categorical_accuracy: 0.5066
Epoch 42/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1153 - sparse_categorical_accuracy: 0.5051 - val_loss: 1.2123 - val_sparse_categorical_accuracy: 0.5109
Epoch 43/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1136 - sparse_categorical_accuracy: 0.5066 - val_loss: 1.2101 - val_sparse_categorical_accuracy: 0.5066
Epoch 44/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1123 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.2077 - val_sparse_categorical_accuracy: 0.5109
Epoch 45/50
2/2 [==============================] - 0s 16ms/step - loss: 1.1114 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.2067 - val_sparse_categorical_accuracy: 0.5240
Epoch 46/50
2/2 [==============================] - 0s 15ms/step - loss: 1.1106 - sparse_categorical_accuracy: 0.5109 - val_loss: 1.2084 - val_sparse_categorical_accuracy: 0.5109
Epoch 47/50
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 1.1091 - sparse_categorical_accuracy: 0.5080 - val_loss: 1.2129 - val_sparse_categorical_accuracy: 0.5109
Epoch 48/50
2/2 [==============================] - 0s 19ms/step - loss: 1.1092 - sparse_categorical_accuracy: 0.5226 - val_loss: 1.2148 - val_sparse_categorical_accuracy: 0.4978
Epoch 49/50
2/2 [==============================] - 0s 15ms/step - loss: 1.1085 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.2103 - val_sparse_categorical_accuracy: 0.5153
Epoch 50/50
2/2 [==============================] - 0s 15ms/step - loss: 1.1071 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.2043 - val_sparse_categorical_accuracy: 0.5153
</pre>
<pre>
<keras.callbacks.History at 0x7f7b18088490>
</pre>

```python
model.fit(X_tn,y_tn, validation_split = 0.25, epochs = 2000, batch_size = 500)
```

<pre>
Epoch 1/2000
2/2 [==============================] - 0s 41ms/step - loss: 1.1053 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1995 - val_sparse_categorical_accuracy: 0.5109
Epoch 2/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.1052 - sparse_categorical_accuracy: 0.5109 - val_loss: 1.1954 - val_sparse_categorical_accuracy: 0.5153
Epoch 3/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.1047 - sparse_categorical_accuracy: 0.5080 - val_loss: 1.1943 - val_sparse_categorical_accuracy: 0.5066
Epoch 4/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.1044 - sparse_categorical_accuracy: 0.5124 - val_loss: 1.1936 - val_sparse_categorical_accuracy: 0.4934
Epoch 5/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.1036 - sparse_categorical_accuracy: 0.5226 - val_loss: 1.1947 - val_sparse_categorical_accuracy: 0.4934
Epoch 6/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.1025 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1953 - val_sparse_categorical_accuracy: 0.4978
Epoch 7/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.1016 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1946 - val_sparse_categorical_accuracy: 0.4978
Epoch 8/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.1007 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1932 - val_sparse_categorical_accuracy: 0.5153
Epoch 9/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0998 - sparse_categorical_accuracy: 0.5124 - val_loss: 1.1926 - val_sparse_categorical_accuracy: 0.5153
Epoch 10/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0993 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1951 - val_sparse_categorical_accuracy: 0.5284
Epoch 11/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0985 - sparse_categorical_accuracy: 0.5124 - val_loss: 1.1974 - val_sparse_categorical_accuracy: 0.5197
Epoch 12/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0979 - sparse_categorical_accuracy: 0.5168 - val_loss: 1.1965 - val_sparse_categorical_accuracy: 0.5153
Epoch 13/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0972 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1929 - val_sparse_categorical_accuracy: 0.5197
Epoch 14/2000
2/2 [==============================] - 0s 21ms/step - loss: 1.0968 - sparse_categorical_accuracy: 0.5182 - val_loss: 1.1901 - val_sparse_categorical_accuracy: 0.5240
Epoch 15/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0961 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1903 - val_sparse_categorical_accuracy: 0.5197
Epoch 16/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0955 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1913 - val_sparse_categorical_accuracy: 0.5197
Epoch 17/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0949 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1923 - val_sparse_categorical_accuracy: 0.5197
Epoch 18/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0950 - sparse_categorical_accuracy: 0.5168 - val_loss: 1.1929 - val_sparse_categorical_accuracy: 0.5066
Epoch 19/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0942 - sparse_categorical_accuracy: 0.5182 - val_loss: 1.1937 - val_sparse_categorical_accuracy: 0.4803
Epoch 20/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0942 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1925 - val_sparse_categorical_accuracy: 0.4891
Epoch 21/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0934 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1894 - val_sparse_categorical_accuracy: 0.5066
Epoch 22/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0926 - sparse_categorical_accuracy: 0.5095 - val_loss: 1.1855 - val_sparse_categorical_accuracy: 0.5153
Epoch 23/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0919 - sparse_categorical_accuracy: 0.5182 - val_loss: 1.1854 - val_sparse_categorical_accuracy: 0.5197
Epoch 24/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0916 - sparse_categorical_accuracy: 0.5168 - val_loss: 1.1885 - val_sparse_categorical_accuracy: 0.5109
Epoch 25/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0908 - sparse_categorical_accuracy: 0.5168 - val_loss: 1.1924 - val_sparse_categorical_accuracy: 0.5066
Epoch 26/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0907 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1961 - val_sparse_categorical_accuracy: 0.4934
Epoch 27/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0915 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1960 - val_sparse_categorical_accuracy: 0.4978
Epoch 28/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0910 - sparse_categorical_accuracy: 0.5241 - val_loss: 1.1912 - val_sparse_categorical_accuracy: 0.5153
Epoch 29/2000
2/2 [==============================] - 0s 18ms/step - loss: 1.0894 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1867 - val_sparse_categorical_accuracy: 0.5109
Epoch 30/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0888 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1811 - val_sparse_categorical_accuracy: 0.5197
Epoch 31/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0888 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1758 - val_sparse_categorical_accuracy: 0.5109
Epoch 32/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0900 - sparse_categorical_accuracy: 0.4964 - val_loss: 1.1737 - val_sparse_categorical_accuracy: 0.5066
Epoch 33/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0906 - sparse_categorical_accuracy: 0.4949 - val_loss: 1.1753 - val_sparse_categorical_accuracy: 0.5240
Epoch 34/2000
2/2 [==============================] - 0s 20ms/step - loss: 1.0889 - sparse_categorical_accuracy: 0.4949 - val_loss: 1.1806 - val_sparse_categorical_accuracy: 0.5328
Epoch 35/2000
2/2 [==============================] - 0s 20ms/step - loss: 1.0877 - sparse_categorical_accuracy: 0.4993 - val_loss: 1.1868 - val_sparse_categorical_accuracy: 0.5197
Epoch 36/2000
2/2 [==============================] - 0s 20ms/step - loss: 1.0874 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1909 - val_sparse_categorical_accuracy: 0.5109
Epoch 37/2000
2/2 [==============================] - 0s 20ms/step - loss: 1.0875 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1918 - val_sparse_categorical_accuracy: 0.5066
Epoch 38/2000
2/2 [==============================] - 0s 19ms/step - loss: 1.0865 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1881 - val_sparse_categorical_accuracy: 0.5022
Epoch 39/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0854 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1849 - val_sparse_categorical_accuracy: 0.5066
Epoch 40/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0851 - sparse_categorical_accuracy: 0.5226 - val_loss: 1.1818 - val_sparse_categorical_accuracy: 0.5022
Epoch 41/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0855 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1790 - val_sparse_categorical_accuracy: 0.5066
Epoch 42/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0851 - sparse_categorical_accuracy: 0.5241 - val_loss: 1.1798 - val_sparse_categorical_accuracy: 0.4978
Epoch 43/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0840 - sparse_categorical_accuracy: 0.5314 - val_loss: 1.1838 - val_sparse_categorical_accuracy: 0.4672
Epoch 44/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0842 - sparse_categorical_accuracy: 0.5328 - val_loss: 1.1880 - val_sparse_categorical_accuracy: 0.4760
Epoch 45/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0841 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1885 - val_sparse_categorical_accuracy: 0.4934
Epoch 46/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 1.0833 - sparse_categorical_accuracy: 0.5270 - val_loss: 1.1863 - val_sparse_categorical_accuracy: 0.5240
Epoch 47/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0835 - sparse_categorical_accuracy: 0.5139 - val_loss: 1.1859 - val_sparse_categorical_accuracy: 0.5153
Epoch 48/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0845 - sparse_categorical_accuracy: 0.5066 - val_loss: 1.1860 - val_sparse_categorical_accuracy: 0.5153
Epoch 49/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0839 - sparse_categorical_accuracy: 0.5095 - val_loss: 1.1846 - val_sparse_categorical_accuracy: 0.5197
Epoch 50/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0823 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1855 - val_sparse_categorical_accuracy: 0.5197
Epoch 51/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0816 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1883 - val_sparse_categorical_accuracy: 0.4934
Epoch 52/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0824 - sparse_categorical_accuracy: 0.5299 - val_loss: 1.1867 - val_sparse_categorical_accuracy: 0.4934
Epoch 53/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0818 - sparse_categorical_accuracy: 0.5270 - val_loss: 1.1795 - val_sparse_categorical_accuracy: 0.5153
Epoch 54/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0805 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1742 - val_sparse_categorical_accuracy: 0.5240
Epoch 55/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0808 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1719 - val_sparse_categorical_accuracy: 0.5109
Epoch 56/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0815 - sparse_categorical_accuracy: 0.5109 - val_loss: 1.1732 - val_sparse_categorical_accuracy: 0.5153
Epoch 57/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0802 - sparse_categorical_accuracy: 0.5182 - val_loss: 1.1785 - val_sparse_categorical_accuracy: 0.5240
Epoch 58/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0789 - sparse_categorical_accuracy: 0.5182 - val_loss: 1.1861 - val_sparse_categorical_accuracy: 0.5109
Epoch 59/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0797 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1910 - val_sparse_categorical_accuracy: 0.4891
Epoch 60/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0809 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1867 - val_sparse_categorical_accuracy: 0.5022
Epoch 61/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0786 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1793 - val_sparse_categorical_accuracy: 0.5109
Epoch 62/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0772 - sparse_categorical_accuracy: 0.5226 - val_loss: 1.1764 - val_sparse_categorical_accuracy: 0.5197
Epoch 63/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0787 - sparse_categorical_accuracy: 0.5124 - val_loss: 1.1761 - val_sparse_categorical_accuracy: 0.5284
Epoch 64/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0801 - sparse_categorical_accuracy: 0.5139 - val_loss: 1.1761 - val_sparse_categorical_accuracy: 0.5153
Epoch 65/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0779 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1805 - val_sparse_categorical_accuracy: 0.5153
Epoch 66/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0775 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1880 - val_sparse_categorical_accuracy: 0.4978
Epoch 67/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0789 - sparse_categorical_accuracy: 0.5241 - val_loss: 1.1870 - val_sparse_categorical_accuracy: 0.5022
Epoch 68/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0779 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1803 - val_sparse_categorical_accuracy: 0.5197
Epoch 69/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0761 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1762 - val_sparse_categorical_accuracy: 0.5197
Epoch 70/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0768 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1738 - val_sparse_categorical_accuracy: 0.5153
Epoch 71/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0767 - sparse_categorical_accuracy: 0.5124 - val_loss: 1.1744 - val_sparse_categorical_accuracy: 0.5153
Epoch 72/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0758 - sparse_categorical_accuracy: 0.5182 - val_loss: 1.1789 - val_sparse_categorical_accuracy: 0.5153
Epoch 73/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0744 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1794 - val_sparse_categorical_accuracy: 0.4978
Epoch 74/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0745 - sparse_categorical_accuracy: 0.5314 - val_loss: 1.1791 - val_sparse_categorical_accuracy: 0.5022
Epoch 75/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0743 - sparse_categorical_accuracy: 0.5358 - val_loss: 1.1804 - val_sparse_categorical_accuracy: 0.4978
Epoch 76/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0737 - sparse_categorical_accuracy: 0.5314 - val_loss: 1.1813 - val_sparse_categorical_accuracy: 0.5022
Epoch 77/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0740 - sparse_categorical_accuracy: 0.5241 - val_loss: 1.1820 - val_sparse_categorical_accuracy: 0.5066
Epoch 78/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0727 - sparse_categorical_accuracy: 0.5328 - val_loss: 1.1776 - val_sparse_categorical_accuracy: 0.5153
Epoch 79/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0730 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1745 - val_sparse_categorical_accuracy: 0.5284
Epoch 80/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0739 - sparse_categorical_accuracy: 0.5153 - val_loss: 1.1746 - val_sparse_categorical_accuracy: 0.5197
Epoch 81/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0737 - sparse_categorical_accuracy: 0.5109 - val_loss: 1.1775 - val_sparse_categorical_accuracy: 0.5240
Epoch 82/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0718 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1776 - val_sparse_categorical_accuracy: 0.5153
Epoch 83/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0714 - sparse_categorical_accuracy: 0.5299 - val_loss: 1.1754 - val_sparse_categorical_accuracy: 0.5153
Epoch 84/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0711 - sparse_categorical_accuracy: 0.5299 - val_loss: 1.1734 - val_sparse_categorical_accuracy: 0.5153
Epoch 85/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0710 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1734 - val_sparse_categorical_accuracy: 0.5109
Epoch 86/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0706 - sparse_categorical_accuracy: 0.5314 - val_loss: 1.1738 - val_sparse_categorical_accuracy: 0.5022
Epoch 87/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0705 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1737 - val_sparse_categorical_accuracy: 0.5022
Epoch 88/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0704 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1755 - val_sparse_categorical_accuracy: 0.5066
Epoch 89/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0702 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1792 - val_sparse_categorical_accuracy: 0.5109
Epoch 90/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0704 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1817 - val_sparse_categorical_accuracy: 0.5197
Epoch 91/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 1.0706 - sparse_categorical_accuracy: 0.5343 - val_loss: 1.1809 - val_sparse_categorical_accuracy: 0.5153
Epoch 92/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0697 - sparse_categorical_accuracy: 0.5328 - val_loss: 1.1760 - val_sparse_categorical_accuracy: 0.5066
Epoch 93/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0687 - sparse_categorical_accuracy: 0.5401 - val_loss: 1.1718 - val_sparse_categorical_accuracy: 0.4978
Epoch 94/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0695 - sparse_categorical_accuracy: 0.5255 - val_loss: 1.1694 - val_sparse_categorical_accuracy: 0.5022
Epoch 95/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0702 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1689 - val_sparse_categorical_accuracy: 0.5022
Epoch 96/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0702 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1704 - val_sparse_categorical_accuracy: 0.4978
Epoch 97/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0688 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1723 - val_sparse_categorical_accuracy: 0.5066
Epoch 98/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0676 - sparse_categorical_accuracy: 0.5226 - val_loss: 1.1752 - val_sparse_categorical_accuracy: 0.5153
Epoch 99/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0685 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1789 - val_sparse_categorical_accuracy: 0.5240
Epoch 100/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0683 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1822 - val_sparse_categorical_accuracy: 0.5371
Epoch 101/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0686 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1828 - val_sparse_categorical_accuracy: 0.5240
Epoch 102/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0677 - sparse_categorical_accuracy: 0.5372 - val_loss: 1.1770 - val_sparse_categorical_accuracy: 0.5109
Epoch 103/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0665 - sparse_categorical_accuracy: 0.5401 - val_loss: 1.1724 - val_sparse_categorical_accuracy: 0.4978
Epoch 104/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0674 - sparse_categorical_accuracy: 0.5431 - val_loss: 1.1714 - val_sparse_categorical_accuracy: 0.4934
Epoch 105/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0670 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1751 - val_sparse_categorical_accuracy: 0.4891
Epoch 106/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0659 - sparse_categorical_accuracy: 0.5431 - val_loss: 1.1778 - val_sparse_categorical_accuracy: 0.4978
Epoch 107/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0656 - sparse_categorical_accuracy: 0.5431 - val_loss: 1.1799 - val_sparse_categorical_accuracy: 0.5066
Epoch 108/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0662 - sparse_categorical_accuracy: 0.5270 - val_loss: 1.1817 - val_sparse_categorical_accuracy: 0.5066
Epoch 109/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0673 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1847 - val_sparse_categorical_accuracy: 0.5066
Epoch 110/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0667 - sparse_categorical_accuracy: 0.5358 - val_loss: 1.1821 - val_sparse_categorical_accuracy: 0.5022
Epoch 111/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0652 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1754 - val_sparse_categorical_accuracy: 0.5022
Epoch 112/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0648 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1707 - val_sparse_categorical_accuracy: 0.5022
Epoch 113/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0654 - sparse_categorical_accuracy: 0.5358 - val_loss: 1.1683 - val_sparse_categorical_accuracy: 0.5022
Epoch 114/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0650 - sparse_categorical_accuracy: 0.5299 - val_loss: 1.1706 - val_sparse_categorical_accuracy: 0.5197
Epoch 115/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0642 - sparse_categorical_accuracy: 0.5299 - val_loss: 1.1751 - val_sparse_categorical_accuracy: 0.5415
Epoch 116/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0646 - sparse_categorical_accuracy: 0.5372 - val_loss: 1.1762 - val_sparse_categorical_accuracy: 0.5459
Epoch 117/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0642 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1727 - val_sparse_categorical_accuracy: 0.5328
Epoch 118/2000
2/2 [==============================] - 0s 17ms/step - loss: 1.0638 - sparse_categorical_accuracy: 0.5226 - val_loss: 1.1705 - val_sparse_categorical_accuracy: 0.5153
Epoch 119/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0637 - sparse_categorical_accuracy: 0.5241 - val_loss: 1.1723 - val_sparse_categorical_accuracy: 0.5022
Epoch 120/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0632 - sparse_categorical_accuracy: 0.5241 - val_loss: 1.1769 - val_sparse_categorical_accuracy: 0.5066
Epoch 121/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0635 - sparse_categorical_accuracy: 0.5328 - val_loss: 1.1817 - val_sparse_categorical_accuracy: 0.4934
Epoch 122/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0625 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1832 - val_sparse_categorical_accuracy: 0.4934
Epoch 123/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0629 - sparse_categorical_accuracy: 0.5460 - val_loss: 1.1853 - val_sparse_categorical_accuracy: 0.4891
Epoch 124/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0639 - sparse_categorical_accuracy: 0.5460 - val_loss: 1.1859 - val_sparse_categorical_accuracy: 0.4891
Epoch 125/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0634 - sparse_categorical_accuracy: 0.5431 - val_loss: 1.1859 - val_sparse_categorical_accuracy: 0.4760
Epoch 126/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0620 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1840 - val_sparse_categorical_accuracy: 0.4672
Epoch 127/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0622 - sparse_categorical_accuracy: 0.5416 - val_loss: 1.1779 - val_sparse_categorical_accuracy: 0.4760
Epoch 128/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0624 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1678 - val_sparse_categorical_accuracy: 0.5109
Epoch 129/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0611 - sparse_categorical_accuracy: 0.5241 - val_loss: 1.1607 - val_sparse_categorical_accuracy: 0.5240
Epoch 130/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0627 - sparse_categorical_accuracy: 0.5197 - val_loss: 1.1608 - val_sparse_categorical_accuracy: 0.5328
Epoch 131/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0618 - sparse_categorical_accuracy: 0.5241 - val_loss: 1.1662 - val_sparse_categorical_accuracy: 0.5371
Epoch 132/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0619 - sparse_categorical_accuracy: 0.5299 - val_loss: 1.1722 - val_sparse_categorical_accuracy: 0.5459
Epoch 133/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0616 - sparse_categorical_accuracy: 0.5416 - val_loss: 1.1708 - val_sparse_categorical_accuracy: 0.5415
Epoch 134/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0604 - sparse_categorical_accuracy: 0.5358 - val_loss: 1.1684 - val_sparse_categorical_accuracy: 0.5197
Epoch 135/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0602 - sparse_categorical_accuracy: 0.5474 - val_loss: 1.1651 - val_sparse_categorical_accuracy: 0.5284
Epoch 136/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 1.0608 - sparse_categorical_accuracy: 0.5328 - val_loss: 1.1641 - val_sparse_categorical_accuracy: 0.5153
Epoch 137/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0608 - sparse_categorical_accuracy: 0.5270 - val_loss: 1.1655 - val_sparse_categorical_accuracy: 0.5109
Epoch 138/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0586 - sparse_categorical_accuracy: 0.5372 - val_loss: 1.1708 - val_sparse_categorical_accuracy: 0.5153
Epoch 139/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0597 - sparse_categorical_accuracy: 0.5401 - val_loss: 1.1732 - val_sparse_categorical_accuracy: 0.5066
Epoch 140/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0593 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1670 - val_sparse_categorical_accuracy: 0.5109
Epoch 141/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0591 - sparse_categorical_accuracy: 0.5314 - val_loss: 1.1629 - val_sparse_categorical_accuracy: 0.4978
Epoch 142/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0590 - sparse_categorical_accuracy: 0.5270 - val_loss: 1.1645 - val_sparse_categorical_accuracy: 0.5022
Epoch 143/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0583 - sparse_categorical_accuracy: 0.5401 - val_loss: 1.1687 - val_sparse_categorical_accuracy: 0.5153
Epoch 144/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0575 - sparse_categorical_accuracy: 0.5460 - val_loss: 1.1706 - val_sparse_categorical_accuracy: 0.5153
Epoch 145/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0565 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1710 - val_sparse_categorical_accuracy: 0.5109
Epoch 146/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0576 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1725 - val_sparse_categorical_accuracy: 0.5153
Epoch 147/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0565 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1739 - val_sparse_categorical_accuracy: 0.4847
Epoch 148/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0560 - sparse_categorical_accuracy: 0.5474 - val_loss: 1.1747 - val_sparse_categorical_accuracy: 0.5066
Epoch 149/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0561 - sparse_categorical_accuracy: 0.5474 - val_loss: 1.1714 - val_sparse_categorical_accuracy: 0.5153
Epoch 150/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0548 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1641 - val_sparse_categorical_accuracy: 0.5197
Epoch 151/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0561 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1621 - val_sparse_categorical_accuracy: 0.5240
Epoch 152/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0559 - sparse_categorical_accuracy: 0.5299 - val_loss: 1.1670 - val_sparse_categorical_accuracy: 0.5240
Epoch 153/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0546 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1780 - val_sparse_categorical_accuracy: 0.4978
Epoch 154/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0559 - sparse_categorical_accuracy: 0.5460 - val_loss: 1.1804 - val_sparse_categorical_accuracy: 0.5022
Epoch 155/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0547 - sparse_categorical_accuracy: 0.5460 - val_loss: 1.1770 - val_sparse_categorical_accuracy: 0.5109
Epoch 156/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0541 - sparse_categorical_accuracy: 0.5372 - val_loss: 1.1768 - val_sparse_categorical_accuracy: 0.5022
Epoch 157/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0546 - sparse_categorical_accuracy: 0.5212 - val_loss: 1.1787 - val_sparse_categorical_accuracy: 0.5066
Epoch 158/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0531 - sparse_categorical_accuracy: 0.5547 - val_loss: 1.1827 - val_sparse_categorical_accuracy: 0.4978
Epoch 159/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0533 - sparse_categorical_accuracy: 0.5474 - val_loss: 1.1793 - val_sparse_categorical_accuracy: 0.5022
Epoch 160/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0537 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1717 - val_sparse_categorical_accuracy: 0.5109
Epoch 161/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0513 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1668 - val_sparse_categorical_accuracy: 0.5022
Epoch 162/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0518 - sparse_categorical_accuracy: 0.5358 - val_loss: 1.1660 - val_sparse_categorical_accuracy: 0.5066
Epoch 163/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0514 - sparse_categorical_accuracy: 0.5416 - val_loss: 1.1704 - val_sparse_categorical_accuracy: 0.5066
Epoch 164/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0509 - sparse_categorical_accuracy: 0.5591 - val_loss: 1.1767 - val_sparse_categorical_accuracy: 0.5022
Epoch 165/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0513 - sparse_categorical_accuracy: 0.5518 - val_loss: 1.1759 - val_sparse_categorical_accuracy: 0.5109
Epoch 166/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0500 - sparse_categorical_accuracy: 0.5620 - val_loss: 1.1725 - val_sparse_categorical_accuracy: 0.5066
Epoch 167/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0506 - sparse_categorical_accuracy: 0.5401 - val_loss: 1.1711 - val_sparse_categorical_accuracy: 0.5066
Epoch 168/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0510 - sparse_categorical_accuracy: 0.5328 - val_loss: 1.1730 - val_sparse_categorical_accuracy: 0.5022
Epoch 169/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0486 - sparse_categorical_accuracy: 0.5562 - val_loss: 1.1799 - val_sparse_categorical_accuracy: 0.4891
Epoch 170/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0502 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1813 - val_sparse_categorical_accuracy: 0.4716
Epoch 171/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0513 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1709 - val_sparse_categorical_accuracy: 0.5240
Epoch 172/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0486 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1620 - val_sparse_categorical_accuracy: 0.5153
Epoch 173/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0491 - sparse_categorical_accuracy: 0.5299 - val_loss: 1.1600 - val_sparse_categorical_accuracy: 0.5153
Epoch 174/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0483 - sparse_categorical_accuracy: 0.5416 - val_loss: 1.1625 - val_sparse_categorical_accuracy: 0.5197
Epoch 175/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0473 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1661 - val_sparse_categorical_accuracy: 0.5153
Epoch 176/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0473 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1695 - val_sparse_categorical_accuracy: 0.5109
Epoch 177/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0467 - sparse_categorical_accuracy: 0.5518 - val_loss: 1.1676 - val_sparse_categorical_accuracy: 0.5197
Epoch 178/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0454 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1650 - val_sparse_categorical_accuracy: 0.5153
Epoch 179/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0492 - sparse_categorical_accuracy: 0.5285 - val_loss: 1.1656 - val_sparse_categorical_accuracy: 0.5109
Epoch 180/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0479 - sparse_categorical_accuracy: 0.5358 - val_loss: 1.1700 - val_sparse_categorical_accuracy: 0.4978
Epoch 181/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 1.0469 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1722 - val_sparse_categorical_accuracy: 0.5109
Epoch 182/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0451 - sparse_categorical_accuracy: 0.5518 - val_loss: 1.1649 - val_sparse_categorical_accuracy: 0.5109
Epoch 183/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0449 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1597 - val_sparse_categorical_accuracy: 0.5109
Epoch 184/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0459 - sparse_categorical_accuracy: 0.5328 - val_loss: 1.1587 - val_sparse_categorical_accuracy: 0.5153
Epoch 185/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0444 - sparse_categorical_accuracy: 0.5358 - val_loss: 1.1590 - val_sparse_categorical_accuracy: 0.5153
Epoch 186/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0438 - sparse_categorical_accuracy: 0.5431 - val_loss: 1.1622 - val_sparse_categorical_accuracy: 0.5153
Epoch 187/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0447 - sparse_categorical_accuracy: 0.5518 - val_loss: 1.1643 - val_sparse_categorical_accuracy: 0.5109
Epoch 188/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0437 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1677 - val_sparse_categorical_accuracy: 0.5153
Epoch 189/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0422 - sparse_categorical_accuracy: 0.5518 - val_loss: 1.1719 - val_sparse_categorical_accuracy: 0.5066
Epoch 190/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0433 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1756 - val_sparse_categorical_accuracy: 0.5109
Epoch 191/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0426 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1825 - val_sparse_categorical_accuracy: 0.5109
Epoch 192/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0448 - sparse_categorical_accuracy: 0.5547 - val_loss: 1.1819 - val_sparse_categorical_accuracy: 0.4934
Epoch 193/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0427 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1652 - val_sparse_categorical_accuracy: 0.5022
Epoch 194/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0407 - sparse_categorical_accuracy: 0.5606 - val_loss: 1.1565 - val_sparse_categorical_accuracy: 0.4934
Epoch 195/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0441 - sparse_categorical_accuracy: 0.5416 - val_loss: 1.1550 - val_sparse_categorical_accuracy: 0.5022
Epoch 196/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0420 - sparse_categorical_accuracy: 0.5358 - val_loss: 1.1647 - val_sparse_categorical_accuracy: 0.5109
Epoch 197/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0404 - sparse_categorical_accuracy: 0.5591 - val_loss: 1.1809 - val_sparse_categorical_accuracy: 0.5022
Epoch 198/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0431 - sparse_categorical_accuracy: 0.5562 - val_loss: 1.1788 - val_sparse_categorical_accuracy: 0.5197
Epoch 199/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0404 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1675 - val_sparse_categorical_accuracy: 0.5153
Epoch 200/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0400 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1612 - val_sparse_categorical_accuracy: 0.4934
Epoch 201/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0409 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1598 - val_sparse_categorical_accuracy: 0.5153
Epoch 202/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0390 - sparse_categorical_accuracy: 0.5591 - val_loss: 1.1620 - val_sparse_categorical_accuracy: 0.4934
Epoch 203/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0386 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1647 - val_sparse_categorical_accuracy: 0.5153
Epoch 204/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0376 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1648 - val_sparse_categorical_accuracy: 0.4978
Epoch 205/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0379 - sparse_categorical_accuracy: 0.5460 - val_loss: 1.1670 - val_sparse_categorical_accuracy: 0.5109
Epoch 206/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0379 - sparse_categorical_accuracy: 0.5445 - val_loss: 1.1707 - val_sparse_categorical_accuracy: 0.4978
Epoch 207/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0382 - sparse_categorical_accuracy: 0.5474 - val_loss: 1.1741 - val_sparse_categorical_accuracy: 0.4978
Epoch 208/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0362 - sparse_categorical_accuracy: 0.5591 - val_loss: 1.1666 - val_sparse_categorical_accuracy: 0.5153
Epoch 209/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0357 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1588 - val_sparse_categorical_accuracy: 0.4978
Epoch 210/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0376 - sparse_categorical_accuracy: 0.5460 - val_loss: 1.1589 - val_sparse_categorical_accuracy: 0.5109
Epoch 211/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0360 - sparse_categorical_accuracy: 0.5577 - val_loss: 1.1655 - val_sparse_categorical_accuracy: 0.4934
Epoch 212/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0365 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1640 - val_sparse_categorical_accuracy: 0.5022
Epoch 213/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0348 - sparse_categorical_accuracy: 0.5577 - val_loss: 1.1593 - val_sparse_categorical_accuracy: 0.5109
Epoch 214/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0335 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1630 - val_sparse_categorical_accuracy: 0.5022
Epoch 215/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0337 - sparse_categorical_accuracy: 0.5547 - val_loss: 1.1647 - val_sparse_categorical_accuracy: 0.5022
Epoch 216/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0319 - sparse_categorical_accuracy: 0.5591 - val_loss: 1.1646 - val_sparse_categorical_accuracy: 0.5022
Epoch 217/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0320 - sparse_categorical_accuracy: 0.5547 - val_loss: 1.1642 - val_sparse_categorical_accuracy: 0.4891
Epoch 218/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0340 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1614 - val_sparse_categorical_accuracy: 0.4934
Epoch 219/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0327 - sparse_categorical_accuracy: 0.5547 - val_loss: 1.1668 - val_sparse_categorical_accuracy: 0.4847
Epoch 220/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0317 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1766 - val_sparse_categorical_accuracy: 0.5022
Epoch 221/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0314 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1842 - val_sparse_categorical_accuracy: 0.4934
Epoch 222/2000
2/2 [==============================] - 0s 19ms/step - loss: 1.0342 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1735 - val_sparse_categorical_accuracy: 0.5240
Epoch 223/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0292 - sparse_categorical_accuracy: 0.5606 - val_loss: 1.1560 - val_sparse_categorical_accuracy: 0.5066
Epoch 224/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0321 - sparse_categorical_accuracy: 0.5401 - val_loss: 1.1505 - val_sparse_categorical_accuracy: 0.4934
Epoch 225/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0323 - sparse_categorical_accuracy: 0.5460 - val_loss: 1.1572 - val_sparse_categorical_accuracy: 0.5109
Epoch 226/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 1.0287 - sparse_categorical_accuracy: 0.5664 - val_loss: 1.1717 - val_sparse_categorical_accuracy: 0.4978
Epoch 227/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0316 - sparse_categorical_accuracy: 0.5620 - val_loss: 1.1712 - val_sparse_categorical_accuracy: 0.5066
Epoch 228/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0288 - sparse_categorical_accuracy: 0.5606 - val_loss: 1.1599 - val_sparse_categorical_accuracy: 0.5022
Epoch 229/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0267 - sparse_categorical_accuracy: 0.5620 - val_loss: 1.1561 - val_sparse_categorical_accuracy: 0.4978
Epoch 230/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0277 - sparse_categorical_accuracy: 0.5547 - val_loss: 1.1595 - val_sparse_categorical_accuracy: 0.5066
Epoch 231/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0255 - sparse_categorical_accuracy: 0.5606 - val_loss: 1.1660 - val_sparse_categorical_accuracy: 0.4978
Epoch 232/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0260 - sparse_categorical_accuracy: 0.5562 - val_loss: 1.1768 - val_sparse_categorical_accuracy: 0.4891
Epoch 233/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0295 - sparse_categorical_accuracy: 0.5489 - val_loss: 1.1697 - val_sparse_categorical_accuracy: 0.5197
Epoch 234/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0255 - sparse_categorical_accuracy: 0.5620 - val_loss: 1.1604 - val_sparse_categorical_accuracy: 0.5109
Epoch 235/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0243 - sparse_categorical_accuracy: 0.5562 - val_loss: 1.1540 - val_sparse_categorical_accuracy: 0.4978
Epoch 236/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0262 - sparse_categorical_accuracy: 0.5533 - val_loss: 1.1596 - val_sparse_categorical_accuracy: 0.5066
Epoch 237/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0232 - sparse_categorical_accuracy: 0.5650 - val_loss: 1.1753 - val_sparse_categorical_accuracy: 0.4847
Epoch 238/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0251 - sparse_categorical_accuracy: 0.5518 - val_loss: 1.1784 - val_sparse_categorical_accuracy: 0.4847
Epoch 239/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0238 - sparse_categorical_accuracy: 0.5577 - val_loss: 1.1656 - val_sparse_categorical_accuracy: 0.4978
Epoch 240/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0221 - sparse_categorical_accuracy: 0.5650 - val_loss: 1.1573 - val_sparse_categorical_accuracy: 0.4891
Epoch 241/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0232 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1590 - val_sparse_categorical_accuracy: 0.5022
Epoch 242/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0209 - sparse_categorical_accuracy: 0.5577 - val_loss: 1.1657 - val_sparse_categorical_accuracy: 0.5066
Epoch 243/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0201 - sparse_categorical_accuracy: 0.5650 - val_loss: 1.1688 - val_sparse_categorical_accuracy: 0.5022
Epoch 244/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0211 - sparse_categorical_accuracy: 0.5620 - val_loss: 1.1627 - val_sparse_categorical_accuracy: 0.5109
Epoch 245/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0193 - sparse_categorical_accuracy: 0.5693 - val_loss: 1.1559 - val_sparse_categorical_accuracy: 0.5109
Epoch 246/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0195 - sparse_categorical_accuracy: 0.5620 - val_loss: 1.1534 - val_sparse_categorical_accuracy: 0.4978
Epoch 247/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0212 - sparse_categorical_accuracy: 0.5504 - val_loss: 1.1558 - val_sparse_categorical_accuracy: 0.5022
Epoch 248/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0194 - sparse_categorical_accuracy: 0.5635 - val_loss: 1.1673 - val_sparse_categorical_accuracy: 0.4978
Epoch 249/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0177 - sparse_categorical_accuracy: 0.5664 - val_loss: 1.1673 - val_sparse_categorical_accuracy: 0.4978
Epoch 250/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0172 - sparse_categorical_accuracy: 0.5708 - val_loss: 1.1627 - val_sparse_categorical_accuracy: 0.5153
Epoch 251/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0174 - sparse_categorical_accuracy: 0.5664 - val_loss: 1.1606 - val_sparse_categorical_accuracy: 0.5109
Epoch 252/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0162 - sparse_categorical_accuracy: 0.5708 - val_loss: 1.1598 - val_sparse_categorical_accuracy: 0.5066
Epoch 253/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0150 - sparse_categorical_accuracy: 0.5781 - val_loss: 1.1593 - val_sparse_categorical_accuracy: 0.4934
Epoch 254/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0155 - sparse_categorical_accuracy: 0.5723 - val_loss: 1.1547 - val_sparse_categorical_accuracy: 0.5109
Epoch 255/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0144 - sparse_categorical_accuracy: 0.5708 - val_loss: 1.1522 - val_sparse_categorical_accuracy: 0.5022
Epoch 256/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0141 - sparse_categorical_accuracy: 0.5664 - val_loss: 1.1577 - val_sparse_categorical_accuracy: 0.5022
Epoch 257/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0125 - sparse_categorical_accuracy: 0.5737 - val_loss: 1.1699 - val_sparse_categorical_accuracy: 0.5022
Epoch 258/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0150 - sparse_categorical_accuracy: 0.5591 - val_loss: 1.1648 - val_sparse_categorical_accuracy: 0.5022
Epoch 259/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0127 - sparse_categorical_accuracy: 0.5737 - val_loss: 1.1519 - val_sparse_categorical_accuracy: 0.5066
Epoch 260/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0121 - sparse_categorical_accuracy: 0.5708 - val_loss: 1.1477 - val_sparse_categorical_accuracy: 0.5153
Epoch 261/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0122 - sparse_categorical_accuracy: 0.5737 - val_loss: 1.1523 - val_sparse_categorical_accuracy: 0.4978
Epoch 262/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0111 - sparse_categorical_accuracy: 0.5752 - val_loss: 1.1521 - val_sparse_categorical_accuracy: 0.4978
Epoch 263/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0108 - sparse_categorical_accuracy: 0.5810 - val_loss: 1.1517 - val_sparse_categorical_accuracy: 0.4934
Epoch 264/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0092 - sparse_categorical_accuracy: 0.5766 - val_loss: 1.1528 - val_sparse_categorical_accuracy: 0.5022
Epoch 265/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0085 - sparse_categorical_accuracy: 0.5781 - val_loss: 1.1492 - val_sparse_categorical_accuracy: 0.5066
Epoch 266/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0090 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1516 - val_sparse_categorical_accuracy: 0.4934
Epoch 267/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0084 - sparse_categorical_accuracy: 0.5825 - val_loss: 1.1603 - val_sparse_categorical_accuracy: 0.4847
Epoch 268/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0073 - sparse_categorical_accuracy: 0.5679 - val_loss: 1.1544 - val_sparse_categorical_accuracy: 0.5022
Epoch 269/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0064 - sparse_categorical_accuracy: 0.5825 - val_loss: 1.1500 - val_sparse_categorical_accuracy: 0.4978
Epoch 270/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0059 - sparse_categorical_accuracy: 0.5737 - val_loss: 1.1480 - val_sparse_categorical_accuracy: 0.5066
Epoch 271/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 1.0058 - sparse_categorical_accuracy: 0.5723 - val_loss: 1.1429 - val_sparse_categorical_accuracy: 0.5066
Epoch 272/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0054 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1356 - val_sparse_categorical_accuracy: 0.5066
Epoch 273/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0045 - sparse_categorical_accuracy: 0.5737 - val_loss: 1.1395 - val_sparse_categorical_accuracy: 0.5066
Epoch 274/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0042 - sparse_categorical_accuracy: 0.5693 - val_loss: 1.1499 - val_sparse_categorical_accuracy: 0.4891
Epoch 275/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0051 - sparse_categorical_accuracy: 0.5752 - val_loss: 1.1573 - val_sparse_categorical_accuracy: 0.4978
Epoch 276/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0037 - sparse_categorical_accuracy: 0.5810 - val_loss: 1.1521 - val_sparse_categorical_accuracy: 0.5066
Epoch 277/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0024 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1540 - val_sparse_categorical_accuracy: 0.5066
Epoch 278/2000
2/2 [==============================] - 0s 16ms/step - loss: 1.0037 - sparse_categorical_accuracy: 0.5708 - val_loss: 1.1543 - val_sparse_categorical_accuracy: 0.4978
Epoch 279/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0017 - sparse_categorical_accuracy: 0.5723 - val_loss: 1.1491 - val_sparse_categorical_accuracy: 0.5066
Epoch 280/2000
2/2 [==============================] - 0s 15ms/step - loss: 1.0010 - sparse_categorical_accuracy: 0.5723 - val_loss: 1.1435 - val_sparse_categorical_accuracy: 0.5109
Epoch 281/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9996 - sparse_categorical_accuracy: 0.5752 - val_loss: 1.1409 - val_sparse_categorical_accuracy: 0.5109
Epoch 282/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9991 - sparse_categorical_accuracy: 0.5781 - val_loss: 1.1395 - val_sparse_categorical_accuracy: 0.5109
Epoch 283/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9989 - sparse_categorical_accuracy: 0.5708 - val_loss: 1.1348 - val_sparse_categorical_accuracy: 0.5066
Epoch 284/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9985 - sparse_categorical_accuracy: 0.5679 - val_loss: 1.1333 - val_sparse_categorical_accuracy: 0.5066
Epoch 285/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9979 - sparse_categorical_accuracy: 0.5752 - val_loss: 1.1368 - val_sparse_categorical_accuracy: 0.4978
Epoch 286/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9965 - sparse_categorical_accuracy: 0.5766 - val_loss: 1.1419 - val_sparse_categorical_accuracy: 0.5022
Epoch 287/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9947 - sparse_categorical_accuracy: 0.5766 - val_loss: 1.1457 - val_sparse_categorical_accuracy: 0.4934
Epoch 288/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9946 - sparse_categorical_accuracy: 0.5781 - val_loss: 1.1465 - val_sparse_categorical_accuracy: 0.4978
Epoch 289/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9949 - sparse_categorical_accuracy: 0.5825 - val_loss: 1.1422 - val_sparse_categorical_accuracy: 0.4978
Epoch 290/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9943 - sparse_categorical_accuracy: 0.5752 - val_loss: 1.1342 - val_sparse_categorical_accuracy: 0.5066
Epoch 291/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9953 - sparse_categorical_accuracy: 0.5752 - val_loss: 1.1315 - val_sparse_categorical_accuracy: 0.5022
Epoch 292/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9939 - sparse_categorical_accuracy: 0.5766 - val_loss: 1.1378 - val_sparse_categorical_accuracy: 0.4978
Epoch 293/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9912 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1472 - val_sparse_categorical_accuracy: 0.5066
Epoch 294/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9915 - sparse_categorical_accuracy: 0.5781 - val_loss: 1.1409 - val_sparse_categorical_accuracy: 0.4978
Epoch 295/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9902 - sparse_categorical_accuracy: 0.5781 - val_loss: 1.1289 - val_sparse_categorical_accuracy: 0.5109
Epoch 296/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9913 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1310 - val_sparse_categorical_accuracy: 0.5022
Epoch 297/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9901 - sparse_categorical_accuracy: 0.5810 - val_loss: 1.1429 - val_sparse_categorical_accuracy: 0.4978
Epoch 298/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9879 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1430 - val_sparse_categorical_accuracy: 0.4934
Epoch 299/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9885 - sparse_categorical_accuracy: 0.5752 - val_loss: 1.1396 - val_sparse_categorical_accuracy: 0.4847
Epoch 300/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9880 - sparse_categorical_accuracy: 0.5766 - val_loss: 1.1421 - val_sparse_categorical_accuracy: 0.4934
Epoch 301/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9869 - sparse_categorical_accuracy: 0.5708 - val_loss: 1.1360 - val_sparse_categorical_accuracy: 0.4978
Epoch 302/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9853 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1322 - val_sparse_categorical_accuracy: 0.4978
Epoch 303/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9849 - sparse_categorical_accuracy: 0.5825 - val_loss: 1.1337 - val_sparse_categorical_accuracy: 0.5109
Epoch 304/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9864 - sparse_categorical_accuracy: 0.5869 - val_loss: 1.1385 - val_sparse_categorical_accuracy: 0.4934
Epoch 305/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9861 - sparse_categorical_accuracy: 0.5883 - val_loss: 1.1328 - val_sparse_categorical_accuracy: 0.5240
Epoch 306/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9828 - sparse_categorical_accuracy: 0.5810 - val_loss: 1.1342 - val_sparse_categorical_accuracy: 0.5022
Epoch 307/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9837 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1417 - val_sparse_categorical_accuracy: 0.5022
Epoch 308/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9816 - sparse_categorical_accuracy: 0.5766 - val_loss: 1.1431 - val_sparse_categorical_accuracy: 0.5022
Epoch 309/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9814 - sparse_categorical_accuracy: 0.5839 - val_loss: 1.1389 - val_sparse_categorical_accuracy: 0.4978
Epoch 310/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9815 - sparse_categorical_accuracy: 0.5839 - val_loss: 1.1272 - val_sparse_categorical_accuracy: 0.5240
Epoch 311/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9793 - sparse_categorical_accuracy: 0.5912 - val_loss: 1.1322 - val_sparse_categorical_accuracy: 0.5153
Epoch 312/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9770 - sparse_categorical_accuracy: 0.5825 - val_loss: 1.1468 - val_sparse_categorical_accuracy: 0.4934
Epoch 313/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9801 - sparse_categorical_accuracy: 0.5752 - val_loss: 1.1471 - val_sparse_categorical_accuracy: 0.4891
Epoch 314/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9782 - sparse_categorical_accuracy: 0.5796 - val_loss: 1.1308 - val_sparse_categorical_accuracy: 0.5153
Epoch 315/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9783 - sparse_categorical_accuracy: 0.5825 - val_loss: 1.1279 - val_sparse_categorical_accuracy: 0.5240
Epoch 316/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.9758 - sparse_categorical_accuracy: 0.5942 - val_loss: 1.1409 - val_sparse_categorical_accuracy: 0.5066
Epoch 317/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9779 - sparse_categorical_accuracy: 0.5883 - val_loss: 1.1290 - val_sparse_categorical_accuracy: 0.5197
Epoch 318/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9741 - sparse_categorical_accuracy: 0.5942 - val_loss: 1.1146 - val_sparse_categorical_accuracy: 0.5109
Epoch 319/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9777 - sparse_categorical_accuracy: 0.5971 - val_loss: 1.1190 - val_sparse_categorical_accuracy: 0.5197
Epoch 320/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9721 - sparse_categorical_accuracy: 0.5985 - val_loss: 1.1453 - val_sparse_categorical_accuracy: 0.5240
Epoch 321/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9755 - sparse_categorical_accuracy: 0.5942 - val_loss: 1.1562 - val_sparse_categorical_accuracy: 0.5415
Epoch 322/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9762 - sparse_categorical_accuracy: 0.5971 - val_loss: 1.1244 - val_sparse_categorical_accuracy: 0.5153
Epoch 323/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9718 - sparse_categorical_accuracy: 0.5985 - val_loss: 1.1150 - val_sparse_categorical_accuracy: 0.5109
Epoch 324/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9742 - sparse_categorical_accuracy: 0.5869 - val_loss: 1.1311 - val_sparse_categorical_accuracy: 0.5066
Epoch 325/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9704 - sparse_categorical_accuracy: 0.5927 - val_loss: 1.1442 - val_sparse_categorical_accuracy: 0.5240
Epoch 326/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9684 - sparse_categorical_accuracy: 0.5985 - val_loss: 1.1297 - val_sparse_categorical_accuracy: 0.5371
Epoch 327/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9682 - sparse_categorical_accuracy: 0.6015 - val_loss: 1.1176 - val_sparse_categorical_accuracy: 0.5328
Epoch 328/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9697 - sparse_categorical_accuracy: 0.5927 - val_loss: 1.1215 - val_sparse_categorical_accuracy: 0.5284
Epoch 329/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9672 - sparse_categorical_accuracy: 0.6058 - val_loss: 1.1284 - val_sparse_categorical_accuracy: 0.5197
Epoch 330/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9659 - sparse_categorical_accuracy: 0.5985 - val_loss: 1.1283 - val_sparse_categorical_accuracy: 0.5153
Epoch 331/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9653 - sparse_categorical_accuracy: 0.5985 - val_loss: 1.1289 - val_sparse_categorical_accuracy: 0.5284
Epoch 332/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9630 - sparse_categorical_accuracy: 0.6073 - val_loss: 1.1326 - val_sparse_categorical_accuracy: 0.5415
Epoch 333/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9631 - sparse_categorical_accuracy: 0.6131 - val_loss: 1.1261 - val_sparse_categorical_accuracy: 0.5328
Epoch 334/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9617 - sparse_categorical_accuracy: 0.6146 - val_loss: 1.1159 - val_sparse_categorical_accuracy: 0.5066
Epoch 335/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9634 - sparse_categorical_accuracy: 0.5985 - val_loss: 1.1191 - val_sparse_categorical_accuracy: 0.5153
Epoch 336/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9593 - sparse_categorical_accuracy: 0.6015 - val_loss: 1.1424 - val_sparse_categorical_accuracy: 0.5328
Epoch 337/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9620 - sparse_categorical_accuracy: 0.6131 - val_loss: 1.1369 - val_sparse_categorical_accuracy: 0.5328
Epoch 338/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9593 - sparse_categorical_accuracy: 0.5985 - val_loss: 1.1194 - val_sparse_categorical_accuracy: 0.5371
Epoch 339/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9583 - sparse_categorical_accuracy: 0.6029 - val_loss: 1.1167 - val_sparse_categorical_accuracy: 0.5415
Epoch 340/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9566 - sparse_categorical_accuracy: 0.6117 - val_loss: 1.1267 - val_sparse_categorical_accuracy: 0.5371
Epoch 341/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9564 - sparse_categorical_accuracy: 0.6234 - val_loss: 1.1228 - val_sparse_categorical_accuracy: 0.5459
Epoch 342/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9548 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.1156 - val_sparse_categorical_accuracy: 0.5415
Epoch 343/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9541 - sparse_categorical_accuracy: 0.6044 - val_loss: 1.1166 - val_sparse_categorical_accuracy: 0.5240
Epoch 344/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9535 - sparse_categorical_accuracy: 0.6044 - val_loss: 1.1187 - val_sparse_categorical_accuracy: 0.5197
Epoch 345/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9526 - sparse_categorical_accuracy: 0.6088 - val_loss: 1.1079 - val_sparse_categorical_accuracy: 0.5153
Epoch 346/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9527 - sparse_categorical_accuracy: 0.6088 - val_loss: 1.1055 - val_sparse_categorical_accuracy: 0.5284
Epoch 347/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9516 - sparse_categorical_accuracy: 0.6088 - val_loss: 1.1239 - val_sparse_categorical_accuracy: 0.5415
Epoch 348/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9528 - sparse_categorical_accuracy: 0.6190 - val_loss: 1.1289 - val_sparse_categorical_accuracy: 0.5371
Epoch 349/2000
2/2 [==============================] - 0s 22ms/step - loss: 0.9517 - sparse_categorical_accuracy: 0.6175 - val_loss: 1.1146 - val_sparse_categorical_accuracy: 0.5109
Epoch 350/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9504 - sparse_categorical_accuracy: 0.6029 - val_loss: 1.1087 - val_sparse_categorical_accuracy: 0.5197
Epoch 351/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9502 - sparse_categorical_accuracy: 0.6058 - val_loss: 1.1066 - val_sparse_categorical_accuracy: 0.5328
Epoch 352/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9476 - sparse_categorical_accuracy: 0.6088 - val_loss: 1.1077 - val_sparse_categorical_accuracy: 0.5328
Epoch 353/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9475 - sparse_categorical_accuracy: 0.6234 - val_loss: 1.1065 - val_sparse_categorical_accuracy: 0.5371
Epoch 354/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9472 - sparse_categorical_accuracy: 0.6204 - val_loss: 1.1013 - val_sparse_categorical_accuracy: 0.5328
Epoch 355/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9449 - sparse_categorical_accuracy: 0.6161 - val_loss: 1.1024 - val_sparse_categorical_accuracy: 0.5371
Epoch 356/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9449 - sparse_categorical_accuracy: 0.6190 - val_loss: 1.1114 - val_sparse_categorical_accuracy: 0.5459
Epoch 357/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9452 - sparse_categorical_accuracy: 0.6088 - val_loss: 1.1192 - val_sparse_categorical_accuracy: 0.5371
Epoch 358/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9441 - sparse_categorical_accuracy: 0.6161 - val_loss: 1.1084 - val_sparse_categorical_accuracy: 0.5415
Epoch 359/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9430 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.0989 - val_sparse_categorical_accuracy: 0.5415
Epoch 360/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9422 - sparse_categorical_accuracy: 0.6292 - val_loss: 1.1038 - val_sparse_categorical_accuracy: 0.5371
Epoch 361/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.9420 - sparse_categorical_accuracy: 0.6336 - val_loss: 1.0989 - val_sparse_categorical_accuracy: 0.5284
Epoch 362/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9427 - sparse_categorical_accuracy: 0.6336 - val_loss: 1.1074 - val_sparse_categorical_accuracy: 0.5415
Epoch 363/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9395 - sparse_categorical_accuracy: 0.6263 - val_loss: 1.1296 - val_sparse_categorical_accuracy: 0.5459
Epoch 364/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9425 - sparse_categorical_accuracy: 0.6248 - val_loss: 1.1140 - val_sparse_categorical_accuracy: 0.5459
Epoch 365/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9383 - sparse_categorical_accuracy: 0.6277 - val_loss: 1.0964 - val_sparse_categorical_accuracy: 0.5328
Epoch 366/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9396 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.0924 - val_sparse_categorical_accuracy: 0.5328
Epoch 367/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9384 - sparse_categorical_accuracy: 0.6277 - val_loss: 1.0906 - val_sparse_categorical_accuracy: 0.5502
Epoch 368/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9375 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.0940 - val_sparse_categorical_accuracy: 0.5459
Epoch 369/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9365 - sparse_categorical_accuracy: 0.6263 - val_loss: 1.1018 - val_sparse_categorical_accuracy: 0.5415
Epoch 370/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9345 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.1011 - val_sparse_categorical_accuracy: 0.5284
Epoch 371/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9358 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.1022 - val_sparse_categorical_accuracy: 0.5371
Epoch 372/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9361 - sparse_categorical_accuracy: 0.6204 - val_loss: 1.1179 - val_sparse_categorical_accuracy: 0.5502
Epoch 373/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9345 - sparse_categorical_accuracy: 0.6204 - val_loss: 1.1195 - val_sparse_categorical_accuracy: 0.5546
Epoch 374/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9336 - sparse_categorical_accuracy: 0.6263 - val_loss: 1.1000 - val_sparse_categorical_accuracy: 0.5371
Epoch 375/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9325 - sparse_categorical_accuracy: 0.6234 - val_loss: 1.0889 - val_sparse_categorical_accuracy: 0.5371
Epoch 376/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9358 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.1012 - val_sparse_categorical_accuracy: 0.5459
Epoch 377/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9284 - sparse_categorical_accuracy: 0.6350 - val_loss: 1.1168 - val_sparse_categorical_accuracy: 0.5502
Epoch 378/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9316 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.1072 - val_sparse_categorical_accuracy: 0.5546
Epoch 379/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9284 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.0894 - val_sparse_categorical_accuracy: 0.5415
Epoch 380/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9280 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.0919 - val_sparse_categorical_accuracy: 0.5415
Epoch 381/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9274 - sparse_categorical_accuracy: 0.6336 - val_loss: 1.1008 - val_sparse_categorical_accuracy: 0.5415
Epoch 382/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9258 - sparse_categorical_accuracy: 0.6336 - val_loss: 1.0961 - val_sparse_categorical_accuracy: 0.5415
Epoch 383/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9251 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.0948 - val_sparse_categorical_accuracy: 0.5546
Epoch 384/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9249 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0932 - val_sparse_categorical_accuracy: 0.5415
Epoch 385/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9239 - sparse_categorical_accuracy: 0.6336 - val_loss: 1.0823 - val_sparse_categorical_accuracy: 0.5371
Epoch 386/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9244 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0938 - val_sparse_categorical_accuracy: 0.5502
Epoch 387/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9260 - sparse_categorical_accuracy: 0.6248 - val_loss: 1.0961 - val_sparse_categorical_accuracy: 0.5459
Epoch 388/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9217 - sparse_categorical_accuracy: 0.6350 - val_loss: 1.0784 - val_sparse_categorical_accuracy: 0.5459
Epoch 389/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9226 - sparse_categorical_accuracy: 0.6350 - val_loss: 1.0907 - val_sparse_categorical_accuracy: 0.5502
Epoch 390/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9200 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.1227 - val_sparse_categorical_accuracy: 0.5240
Epoch 391/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9249 - sparse_categorical_accuracy: 0.6234 - val_loss: 1.1016 - val_sparse_categorical_accuracy: 0.5459
Epoch 392/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9199 - sparse_categorical_accuracy: 0.6292 - val_loss: 1.0810 - val_sparse_categorical_accuracy: 0.5371
Epoch 393/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9259 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0952 - val_sparse_categorical_accuracy: 0.5328
Epoch 394/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9189 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.1186 - val_sparse_categorical_accuracy: 0.5459
Epoch 395/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9207 - sparse_categorical_accuracy: 0.6204 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5633
Epoch 396/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9181 - sparse_categorical_accuracy: 0.6292 - val_loss: 1.0707 - val_sparse_categorical_accuracy: 0.5546
Epoch 397/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9234 - sparse_categorical_accuracy: 0.6219 - val_loss: 1.0872 - val_sparse_categorical_accuracy: 0.5459
Epoch 398/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9155 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.1076 - val_sparse_categorical_accuracy: 0.5546
Epoch 399/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9182 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.0831 - val_sparse_categorical_accuracy: 0.5328
Epoch 400/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9176 - sparse_categorical_accuracy: 0.6350 - val_loss: 1.0741 - val_sparse_categorical_accuracy: 0.5415
Epoch 401/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9158 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0990 - val_sparse_categorical_accuracy: 0.5415
Epoch 402/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9165 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.1049 - val_sparse_categorical_accuracy: 0.5502
Epoch 403/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9137 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0792 - val_sparse_categorical_accuracy: 0.5502
Epoch 404/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9158 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.0767 - val_sparse_categorical_accuracy: 0.5459
Epoch 405/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9117 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.1083 - val_sparse_categorical_accuracy: 0.5371
Epoch 406/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.9169 - sparse_categorical_accuracy: 0.6292 - val_loss: 1.0992 - val_sparse_categorical_accuracy: 0.5415
Epoch 407/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9113 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0659 - val_sparse_categorical_accuracy: 0.5502
Epoch 408/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9146 - sparse_categorical_accuracy: 0.6336 - val_loss: 1.0700 - val_sparse_categorical_accuracy: 0.5459
Epoch 409/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9088 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0977 - val_sparse_categorical_accuracy: 0.5459
Epoch 410/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9134 - sparse_categorical_accuracy: 0.6321 - val_loss: 1.0924 - val_sparse_categorical_accuracy: 0.5633
Epoch 411/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9098 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0669 - val_sparse_categorical_accuracy: 0.5502
Epoch 412/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9103 - sparse_categorical_accuracy: 0.6277 - val_loss: 1.0766 - val_sparse_categorical_accuracy: 0.5546
Epoch 413/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9078 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.1005 - val_sparse_categorical_accuracy: 0.5415
Epoch 414/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9082 - sparse_categorical_accuracy: 0.6438 - val_loss: 1.0865 - val_sparse_categorical_accuracy: 0.5459
Epoch 415/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9048 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0777 - val_sparse_categorical_accuracy: 0.5546
Epoch 416/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9088 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0921 - val_sparse_categorical_accuracy: 0.5459
Epoch 417/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9089 - sparse_categorical_accuracy: 0.6263 - val_loss: 1.1059 - val_sparse_categorical_accuracy: 0.5633
Epoch 418/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9039 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0768 - val_sparse_categorical_accuracy: 0.5459
Epoch 419/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9070 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0718 - val_sparse_categorical_accuracy: 0.5328
Epoch 420/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9058 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.0932 - val_sparse_categorical_accuracy: 0.5677
Epoch 421/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9052 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0868 - val_sparse_categorical_accuracy: 0.5546
Epoch 422/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9053 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.0714 - val_sparse_categorical_accuracy: 0.5546
Epoch 423/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9011 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0975 - val_sparse_categorical_accuracy: 0.5546
Epoch 424/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9023 - sparse_categorical_accuracy: 0.6292 - val_loss: 1.1049 - val_sparse_categorical_accuracy: 0.5590
Epoch 425/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9020 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0747 - val_sparse_categorical_accuracy: 0.5546
Epoch 426/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9021 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0705 - val_sparse_categorical_accuracy: 0.5590
Epoch 427/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9014 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0966 - val_sparse_categorical_accuracy: 0.5415
Epoch 428/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9018 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0945 - val_sparse_categorical_accuracy: 0.5459
Epoch 429/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8959 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0646 - val_sparse_categorical_accuracy: 0.5590
Epoch 430/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9052 - sparse_categorical_accuracy: 0.6321 - val_loss: 1.0642 - val_sparse_categorical_accuracy: 0.5546
Epoch 431/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9009 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.1009 - val_sparse_categorical_accuracy: 0.5546
Epoch 432/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9009 - sparse_categorical_accuracy: 0.6321 - val_loss: 1.0938 - val_sparse_categorical_accuracy: 0.5502
Epoch 433/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8962 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0604 - val_sparse_categorical_accuracy: 0.5546
Epoch 434/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9022 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0633 - val_sparse_categorical_accuracy: 0.5415
Epoch 435/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8991 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.1144 - val_sparse_categorical_accuracy: 0.5197
Epoch 436/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.9028 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.1151 - val_sparse_categorical_accuracy: 0.5240
Epoch 437/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8974 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0646 - val_sparse_categorical_accuracy: 0.5590
Epoch 438/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8999 - sparse_categorical_accuracy: 0.6263 - val_loss: 1.0536 - val_sparse_categorical_accuracy: 0.5546
Epoch 439/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9012 - sparse_categorical_accuracy: 0.6336 - val_loss: 1.0895 - val_sparse_categorical_accuracy: 0.5546
Epoch 440/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9037 - sparse_categorical_accuracy: 0.6350 - val_loss: 1.0989 - val_sparse_categorical_accuracy: 0.5546
Epoch 441/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8969 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0585 - val_sparse_categorical_accuracy: 0.5415
Epoch 442/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8995 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0680 - val_sparse_categorical_accuracy: 0.5546
Epoch 443/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8944 - sparse_categorical_accuracy: 0.6350 - val_loss: 1.1059 - val_sparse_categorical_accuracy: 0.5371
Epoch 444/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8974 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0750 - val_sparse_categorical_accuracy: 0.5590
Epoch 445/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8878 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0541 - val_sparse_categorical_accuracy: 0.5459
Epoch 446/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9040 - sparse_categorical_accuracy: 0.6321 - val_loss: 1.0693 - val_sparse_categorical_accuracy: 0.5546
Epoch 447/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8950 - sparse_categorical_accuracy: 0.6292 - val_loss: 1.1234 - val_sparse_categorical_accuracy: 0.5022
Epoch 448/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.9022 - sparse_categorical_accuracy: 0.6307 - val_loss: 1.0686 - val_sparse_categorical_accuracy: 0.5677
Epoch 449/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8940 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0567 - val_sparse_categorical_accuracy: 0.5590
Epoch 450/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8966 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0961 - val_sparse_categorical_accuracy: 0.5459
Epoch 451/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.8955 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.1338 - val_sparse_categorical_accuracy: 0.5197
Epoch 452/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8998 - sparse_categorical_accuracy: 0.6321 - val_loss: 1.0785 - val_sparse_categorical_accuracy: 0.5590
Epoch 453/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8891 - sparse_categorical_accuracy: 0.6438 - val_loss: 1.0620 - val_sparse_categorical_accuracy: 0.5546
Epoch 454/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8888 - sparse_categorical_accuracy: 0.6438 - val_loss: 1.0914 - val_sparse_categorical_accuracy: 0.5459
Epoch 455/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8942 - sparse_categorical_accuracy: 0.6292 - val_loss: 1.1098 - val_sparse_categorical_accuracy: 0.5197
Epoch 456/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8960 - sparse_categorical_accuracy: 0.6350 - val_loss: 1.0711 - val_sparse_categorical_accuracy: 0.5633
Epoch 457/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8869 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0654 - val_sparse_categorical_accuracy: 0.5502
Epoch 458/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8861 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0826 - val_sparse_categorical_accuracy: 0.5677
Epoch 459/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8833 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0993 - val_sparse_categorical_accuracy: 0.5371
Epoch 460/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8862 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0914 - val_sparse_categorical_accuracy: 0.5502
Epoch 461/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8842 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0760 - val_sparse_categorical_accuracy: 0.5633
Epoch 462/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8826 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.0682 - val_sparse_categorical_accuracy: 0.5633
Epoch 463/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8826 - sparse_categorical_accuracy: 0.6453 - val_loss: 1.0737 - val_sparse_categorical_accuracy: 0.5633
Epoch 464/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8830 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0762 - val_sparse_categorical_accuracy: 0.5633
Epoch 465/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8823 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0681 - val_sparse_categorical_accuracy: 0.5633
Epoch 466/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8802 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.0754 - val_sparse_categorical_accuracy: 0.5677
Epoch 467/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8795 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0766 - val_sparse_categorical_accuracy: 0.5633
Epoch 468/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8790 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0713 - val_sparse_categorical_accuracy: 0.5633
Epoch 469/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8793 - sparse_categorical_accuracy: 0.6350 - val_loss: 1.0690 - val_sparse_categorical_accuracy: 0.5677
Epoch 470/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8796 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0732 - val_sparse_categorical_accuracy: 0.5677
Epoch 471/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8787 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0901 - val_sparse_categorical_accuracy: 0.5546
Epoch 472/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.8808 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0953 - val_sparse_categorical_accuracy: 0.5415
Epoch 473/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8774 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0735 - val_sparse_categorical_accuracy: 0.5371
Epoch 474/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8825 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0749 - val_sparse_categorical_accuracy: 0.5459
Epoch 475/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8786 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.1041 - val_sparse_categorical_accuracy: 0.5415
Epoch 476/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8813 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0923 - val_sparse_categorical_accuracy: 0.5590
Epoch 477/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8777 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.0644 - val_sparse_categorical_accuracy: 0.5633
Epoch 478/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8796 - sparse_categorical_accuracy: 0.6336 - val_loss: 1.0709 - val_sparse_categorical_accuracy: 0.5590
Epoch 479/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8748 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.1004 - val_sparse_categorical_accuracy: 0.5677
Epoch 480/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8798 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0966 - val_sparse_categorical_accuracy: 0.5633
Epoch 481/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8768 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0702 - val_sparse_categorical_accuracy: 0.5677
Epoch 482/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8772 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0672 - val_sparse_categorical_accuracy: 0.5633
Epoch 483/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8772 - sparse_categorical_accuracy: 0.6453 - val_loss: 1.0939 - val_sparse_categorical_accuracy: 0.5633
Epoch 484/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8770 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.1031 - val_sparse_categorical_accuracy: 0.5633
Epoch 485/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8768 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0847 - val_sparse_categorical_accuracy: 0.5546
Epoch 486/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8777 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0852 - val_sparse_categorical_accuracy: 0.5546
Epoch 487/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8752 - sparse_categorical_accuracy: 0.6453 - val_loss: 1.0928 - val_sparse_categorical_accuracy: 0.5721
Epoch 488/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8725 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0780 - val_sparse_categorical_accuracy: 0.5808
Epoch 489/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8720 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0706 - val_sparse_categorical_accuracy: 0.5502
Epoch 490/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8720 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.0831 - val_sparse_categorical_accuracy: 0.5371
Epoch 491/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8746 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0852 - val_sparse_categorical_accuracy: 0.5459
Epoch 492/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8718 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0686 - val_sparse_categorical_accuracy: 0.5633
Epoch 493/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8720 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0643 - val_sparse_categorical_accuracy: 0.5721
Epoch 494/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8701 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0680 - val_sparse_categorical_accuracy: 0.5721
Epoch 495/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8699 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0840 - val_sparse_categorical_accuracy: 0.5371
Epoch 496/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.8700 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0827 - val_sparse_categorical_accuracy: 0.5590
Epoch 497/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8691 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0799 - val_sparse_categorical_accuracy: 0.5546
Epoch 498/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8714 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0826 - val_sparse_categorical_accuracy: 0.5590
Epoch 499/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8693 - sparse_categorical_accuracy: 0.6380 - val_loss: 1.0893 - val_sparse_categorical_accuracy: 0.5721
Epoch 500/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8687 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0811 - val_sparse_categorical_accuracy: 0.5677
Epoch 501/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8693 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0674 - val_sparse_categorical_accuracy: 0.5633
Epoch 502/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8677 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0652 - val_sparse_categorical_accuracy: 0.5546
Epoch 503/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8678 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0807 - val_sparse_categorical_accuracy: 0.5590
Epoch 504/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8660 - sparse_categorical_accuracy: 0.6453 - val_loss: 1.1055 - val_sparse_categorical_accuracy: 0.5502
Epoch 505/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8698 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.1088 - val_sparse_categorical_accuracy: 0.5459
Epoch 506/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8733 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0823 - val_sparse_categorical_accuracy: 0.5677
Epoch 507/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8674 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0796 - val_sparse_categorical_accuracy: 0.5677
Epoch 508/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8649 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0706 - val_sparse_categorical_accuracy: 0.5677
Epoch 509/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8656 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0758 - val_sparse_categorical_accuracy: 0.5677
Epoch 510/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8664 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0743 - val_sparse_categorical_accuracy: 0.5633
Epoch 511/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8636 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.0665 - val_sparse_categorical_accuracy: 0.5590
Epoch 512/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8658 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0733 - val_sparse_categorical_accuracy: 0.5633
Epoch 513/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8640 - sparse_categorical_accuracy: 0.6394 - val_loss: 1.0805 - val_sparse_categorical_accuracy: 0.5677
Epoch 514/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8635 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0757 - val_sparse_categorical_accuracy: 0.5808
Epoch 515/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8635 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0668 - val_sparse_categorical_accuracy: 0.5546
Epoch 516/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8635 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0642 - val_sparse_categorical_accuracy: 0.5633
Epoch 517/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8617 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.0761 - val_sparse_categorical_accuracy: 0.5721
Epoch 518/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8644 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0822 - val_sparse_categorical_accuracy: 0.5677
Epoch 519/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8641 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0706 - val_sparse_categorical_accuracy: 0.5590
Epoch 520/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8631 - sparse_categorical_accuracy: 0.6453 - val_loss: 1.0602 - val_sparse_categorical_accuracy: 0.5502
Epoch 521/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8637 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.0790 - val_sparse_categorical_accuracy: 0.5721
Epoch 522/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8631 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0862 - val_sparse_categorical_accuracy: 0.5590
Epoch 523/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8623 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0699 - val_sparse_categorical_accuracy: 0.5764
Epoch 524/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8614 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0753 - val_sparse_categorical_accuracy: 0.5633
Epoch 525/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8599 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0987 - val_sparse_categorical_accuracy: 0.5371
Epoch 526/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8632 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0958 - val_sparse_categorical_accuracy: 0.5546
Epoch 527/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8620 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0725 - val_sparse_categorical_accuracy: 0.5590
Epoch 528/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8632 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0781 - val_sparse_categorical_accuracy: 0.5677
Epoch 529/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8646 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0865 - val_sparse_categorical_accuracy: 0.5633
Epoch 530/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8602 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0549 - val_sparse_categorical_accuracy: 0.5502
Epoch 531/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8638 - sparse_categorical_accuracy: 0.6453 - val_loss: 1.0589 - val_sparse_categorical_accuracy: 0.5677
Epoch 532/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8591 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.1048 - val_sparse_categorical_accuracy: 0.5415
Epoch 533/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8652 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0955 - val_sparse_categorical_accuracy: 0.5590
Epoch 534/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8617 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0690 - val_sparse_categorical_accuracy: 0.5633
Epoch 535/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8596 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0724 - val_sparse_categorical_accuracy: 0.5677
Epoch 536/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8567 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0922 - val_sparse_categorical_accuracy: 0.5590
Epoch 537/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8626 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0854 - val_sparse_categorical_accuracy: 0.5633
Epoch 538/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8574 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.0612 - val_sparse_categorical_accuracy: 0.5502
Epoch 539/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8604 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.0725 - val_sparse_categorical_accuracy: 0.5633
Epoch 540/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8575 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.1017 - val_sparse_categorical_accuracy: 0.5546
Epoch 541/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.8605 - sparse_categorical_accuracy: 0.6569 - val_loss: 1.0945 - val_sparse_categorical_accuracy: 0.5633
Epoch 542/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8564 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0675 - val_sparse_categorical_accuracy: 0.5633
Epoch 543/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8587 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0668 - val_sparse_categorical_accuracy: 0.5808
Epoch 544/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8595 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0789 - val_sparse_categorical_accuracy: 0.5677
Epoch 545/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8577 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0784 - val_sparse_categorical_accuracy: 0.5546
Epoch 546/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8551 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0587 - val_sparse_categorical_accuracy: 0.5546
Epoch 547/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8600 - sparse_categorical_accuracy: 0.6453 - val_loss: 1.0673 - val_sparse_categorical_accuracy: 0.5721
Epoch 548/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8530 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.1104 - val_sparse_categorical_accuracy: 0.5328
Epoch 549/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8688 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0867 - val_sparse_categorical_accuracy: 0.5633
Epoch 550/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8549 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0622 - val_sparse_categorical_accuracy: 0.5546
Epoch 551/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8667 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0655 - val_sparse_categorical_accuracy: 0.5371
Epoch 552/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8566 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.1087 - val_sparse_categorical_accuracy: 0.5371
Epoch 553/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8655 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0995 - val_sparse_categorical_accuracy: 0.5546
Epoch 554/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8619 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0639 - val_sparse_categorical_accuracy: 0.5546
Epoch 555/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8614 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0741 - val_sparse_categorical_accuracy: 0.5633
Epoch 556/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8574 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.1012 - val_sparse_categorical_accuracy: 0.5633
Epoch 557/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8569 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0843 - val_sparse_categorical_accuracy: 0.5502
Epoch 558/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8557 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0617 - val_sparse_categorical_accuracy: 0.5546
Epoch 559/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8605 - sparse_categorical_accuracy: 0.6438 - val_loss: 1.0583 - val_sparse_categorical_accuracy: 0.5590
Epoch 560/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8531 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0801 - val_sparse_categorical_accuracy: 0.5677
Epoch 561/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8577 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0726 - val_sparse_categorical_accuracy: 0.5633
Epoch 562/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8565 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0567 - val_sparse_categorical_accuracy: 0.5633
Epoch 563/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8524 - sparse_categorical_accuracy: 0.6423 - val_loss: 1.0833 - val_sparse_categorical_accuracy: 0.5633
Epoch 564/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8554 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0907 - val_sparse_categorical_accuracy: 0.5546
Epoch 565/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8555 - sparse_categorical_accuracy: 0.6482 - val_loss: 1.0654 - val_sparse_categorical_accuracy: 0.5764
Epoch 566/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8542 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0739 - val_sparse_categorical_accuracy: 0.5633
Epoch 567/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8525 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0942 - val_sparse_categorical_accuracy: 0.5371
Epoch 568/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8546 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0840 - val_sparse_categorical_accuracy: 0.5633
Epoch 569/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8519 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0648 - val_sparse_categorical_accuracy: 0.5764
Epoch 570/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8532 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0691 - val_sparse_categorical_accuracy: 0.5764
Epoch 571/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8504 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.0882 - val_sparse_categorical_accuracy: 0.5546
Epoch 572/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8532 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0894 - val_sparse_categorical_accuracy: 0.5546
Epoch 573/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8510 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0648 - val_sparse_categorical_accuracy: 0.5502
Epoch 574/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8531 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0736 - val_sparse_categorical_accuracy: 0.5546
Epoch 575/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8492 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0959 - val_sparse_categorical_accuracy: 0.5590
Epoch 576/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8518 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0802 - val_sparse_categorical_accuracy: 0.5721
Epoch 577/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8486 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0648 - val_sparse_categorical_accuracy: 0.5677
Epoch 578/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8491 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0778 - val_sparse_categorical_accuracy: 0.5633
Epoch 579/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8488 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0984 - val_sparse_categorical_accuracy: 0.5546
Epoch 580/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8521 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0848 - val_sparse_categorical_accuracy: 0.5546
Epoch 581/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8491 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0677 - val_sparse_categorical_accuracy: 0.5677
Epoch 582/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8472 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0845 - val_sparse_categorical_accuracy: 0.5502
Epoch 583/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8465 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0952 - val_sparse_categorical_accuracy: 0.5502
Epoch 584/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8492 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0631 - val_sparse_categorical_accuracy: 0.5764
Epoch 585/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.8467 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0569 - val_sparse_categorical_accuracy: 0.5721
Epoch 586/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.8530 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0708 - val_sparse_categorical_accuracy: 0.5546
Epoch 587/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8442 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0900 - val_sparse_categorical_accuracy: 0.5590
Epoch 588/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8462 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0817 - val_sparse_categorical_accuracy: 0.5502
Epoch 589/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8453 - sparse_categorical_accuracy: 0.6365 - val_loss: 1.0686 - val_sparse_categorical_accuracy: 0.5590
Epoch 590/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8465 - sparse_categorical_accuracy: 0.6438 - val_loss: 1.0751 - val_sparse_categorical_accuracy: 0.5546
Epoch 591/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8441 - sparse_categorical_accuracy: 0.6467 - val_loss: 1.0838 - val_sparse_categorical_accuracy: 0.5590
Epoch 592/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8451 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0685 - val_sparse_categorical_accuracy: 0.5721
Epoch 593/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8441 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0619 - val_sparse_categorical_accuracy: 0.5633
Epoch 594/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8442 - sparse_categorical_accuracy: 0.6569 - val_loss: 1.0730 - val_sparse_categorical_accuracy: 0.5546
Epoch 595/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8437 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0784 - val_sparse_categorical_accuracy: 0.5590
Epoch 596/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8428 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.0770 - val_sparse_categorical_accuracy: 0.5677
Epoch 597/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8443 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0685 - val_sparse_categorical_accuracy: 0.5721
Epoch 598/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8440 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0644 - val_sparse_categorical_accuracy: 0.5764
Epoch 599/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8470 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.0796 - val_sparse_categorical_accuracy: 0.5721
Epoch 600/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8423 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0988 - val_sparse_categorical_accuracy: 0.5546
Epoch 601/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8465 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0829 - val_sparse_categorical_accuracy: 0.5502
Epoch 602/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8434 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0612 - val_sparse_categorical_accuracy: 0.5677
Epoch 603/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8417 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0533 - val_sparse_categorical_accuracy: 0.5764
Epoch 604/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8434 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0649 - val_sparse_categorical_accuracy: 0.5633
Epoch 605/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8411 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0865 - val_sparse_categorical_accuracy: 0.5590
Epoch 606/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8427 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0876 - val_sparse_categorical_accuracy: 0.5590
Epoch 607/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8449 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0737 - val_sparse_categorical_accuracy: 0.5721
Epoch 608/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8423 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0766 - val_sparse_categorical_accuracy: 0.5764
Epoch 609/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8416 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0842 - val_sparse_categorical_accuracy: 0.5459
Epoch 610/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8433 - sparse_categorical_accuracy: 0.6569 - val_loss: 1.0820 - val_sparse_categorical_accuracy: 0.5590
Epoch 611/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8412 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0829 - val_sparse_categorical_accuracy: 0.5546
Epoch 612/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8400 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0885 - val_sparse_categorical_accuracy: 0.5502
Epoch 613/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8399 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0816 - val_sparse_categorical_accuracy: 0.5633
Epoch 614/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8392 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0677 - val_sparse_categorical_accuracy: 0.5677
Epoch 615/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8410 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.0692 - val_sparse_categorical_accuracy: 0.5677
Epoch 616/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8391 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0951 - val_sparse_categorical_accuracy: 0.5546
Epoch 617/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8433 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0876 - val_sparse_categorical_accuracy: 0.5633
Epoch 618/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8421 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0750 - val_sparse_categorical_accuracy: 0.5590
Epoch 619/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8391 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0777 - val_sparse_categorical_accuracy: 0.5502
Epoch 620/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8376 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0811 - val_sparse_categorical_accuracy: 0.5502
Epoch 621/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8388 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0780 - val_sparse_categorical_accuracy: 0.5590
Epoch 622/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8377 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0724 - val_sparse_categorical_accuracy: 0.5677
Epoch 623/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8372 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0701 - val_sparse_categorical_accuracy: 0.5590
Epoch 624/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8386 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0799 - val_sparse_categorical_accuracy: 0.5546
Epoch 625/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8369 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0821 - val_sparse_categorical_accuracy: 0.5546
Epoch 626/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8385 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0737 - val_sparse_categorical_accuracy: 0.5721
Epoch 627/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8375 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0771 - val_sparse_categorical_accuracy: 0.5677
Epoch 628/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8375 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0940 - val_sparse_categorical_accuracy: 0.5546
Epoch 629/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8406 - sparse_categorical_accuracy: 0.6569 - val_loss: 1.0760 - val_sparse_categorical_accuracy: 0.5546
Epoch 630/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8350 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0624 - val_sparse_categorical_accuracy: 0.5764
Epoch 631/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.8365 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0833 - val_sparse_categorical_accuracy: 0.5459
Epoch 632/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8368 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.1020 - val_sparse_categorical_accuracy: 0.5633
Epoch 633/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8406 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0790 - val_sparse_categorical_accuracy: 0.5590
Epoch 634/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.8354 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0764 - val_sparse_categorical_accuracy: 0.5546
Epoch 635/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8356 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0735 - val_sparse_categorical_accuracy: 0.5721
Epoch 636/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8367 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0706 - val_sparse_categorical_accuracy: 0.5633
Epoch 637/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8342 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0842 - val_sparse_categorical_accuracy: 0.5633
Epoch 638/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8347 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0795 - val_sparse_categorical_accuracy: 0.5633
Epoch 639/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8356 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0692 - val_sparse_categorical_accuracy: 0.5677
Epoch 640/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8349 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0824 - val_sparse_categorical_accuracy: 0.5633
Epoch 641/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8364 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0844 - val_sparse_categorical_accuracy: 0.5546
Epoch 642/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8366 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0644 - val_sparse_categorical_accuracy: 0.5764
Epoch 643/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8363 - sparse_categorical_accuracy: 0.6540 - val_loss: 1.0840 - val_sparse_categorical_accuracy: 0.5677
Epoch 644/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8344 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.1092 - val_sparse_categorical_accuracy: 0.5153
Epoch 645/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8380 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0772 - val_sparse_categorical_accuracy: 0.5808
Epoch 646/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8354 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0679 - val_sparse_categorical_accuracy: 0.5721
Epoch 647/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8323 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.1005 - val_sparse_categorical_accuracy: 0.5633
Epoch 648/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8388 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0939 - val_sparse_categorical_accuracy: 0.5677
Epoch 649/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8368 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0761 - val_sparse_categorical_accuracy: 0.5677
Epoch 650/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8338 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0870 - val_sparse_categorical_accuracy: 0.5764
Epoch 651/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8341 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0937 - val_sparse_categorical_accuracy: 0.5546
Epoch 652/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8333 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0804 - val_sparse_categorical_accuracy: 0.5677
Epoch 653/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8358 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0754 - val_sparse_categorical_accuracy: 0.5459
Epoch 654/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8378 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0891 - val_sparse_categorical_accuracy: 0.5721
Epoch 655/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8333 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.1024 - val_sparse_categorical_accuracy: 0.5677
Epoch 656/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8317 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0730 - val_sparse_categorical_accuracy: 0.5721
Epoch 657/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8344 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0650 - val_sparse_categorical_accuracy: 0.5590
Epoch 658/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8356 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0924 - val_sparse_categorical_accuracy: 0.5633
Epoch 659/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8341 - sparse_categorical_accuracy: 0.6569 - val_loss: 1.1005 - val_sparse_categorical_accuracy: 0.5633
Epoch 660/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8331 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0755 - val_sparse_categorical_accuracy: 0.5764
Epoch 661/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8325 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0857 - val_sparse_categorical_accuracy: 0.5633
Epoch 662/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8334 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.1011 - val_sparse_categorical_accuracy: 0.5415
Epoch 663/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8309 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0798 - val_sparse_categorical_accuracy: 0.5852
Epoch 664/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8315 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0750 - val_sparse_categorical_accuracy: 0.5808
Epoch 665/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8300 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0853 - val_sparse_categorical_accuracy: 0.5721
Epoch 666/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8303 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0825 - val_sparse_categorical_accuracy: 0.5764
Epoch 667/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8279 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0793 - val_sparse_categorical_accuracy: 0.5764
Epoch 668/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8295 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0870 - val_sparse_categorical_accuracy: 0.5721
Epoch 669/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8277 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0871 - val_sparse_categorical_accuracy: 0.5852
Epoch 670/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8290 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0727 - val_sparse_categorical_accuracy: 0.5808
Epoch 671/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8291 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0776 - val_sparse_categorical_accuracy: 0.5852
Epoch 672/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8265 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0977 - val_sparse_categorical_accuracy: 0.5502
Epoch 673/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8302 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0844 - val_sparse_categorical_accuracy: 0.5808
Epoch 674/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8257 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0687 - val_sparse_categorical_accuracy: 0.5808
Epoch 675/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8333 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0814 - val_sparse_categorical_accuracy: 0.5721
Epoch 676/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.8289 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.1039 - val_sparse_categorical_accuracy: 0.5677
Epoch 677/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8293 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0848 - val_sparse_categorical_accuracy: 0.5721
Epoch 678/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8262 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0714 - val_sparse_categorical_accuracy: 0.5764
Epoch 679/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8277 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0824 - val_sparse_categorical_accuracy: 0.5721
Epoch 680/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8266 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0977 - val_sparse_categorical_accuracy: 0.5546
Epoch 681/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8323 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0671 - val_sparse_categorical_accuracy: 0.5721
Epoch 682/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8309 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0772 - val_sparse_categorical_accuracy: 0.5808
Epoch 683/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8278 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.1089 - val_sparse_categorical_accuracy: 0.5415
Epoch 684/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8282 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0766 - val_sparse_categorical_accuracy: 0.5546
Epoch 685/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8309 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0685 - val_sparse_categorical_accuracy: 0.5590
Epoch 686/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8336 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.0888 - val_sparse_categorical_accuracy: 0.5633
Epoch 687/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8252 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0928 - val_sparse_categorical_accuracy: 0.5633
Epoch 688/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8296 - sparse_categorical_accuracy: 0.6526 - val_loss: 1.0771 - val_sparse_categorical_accuracy: 0.5895
Epoch 689/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8300 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0761 - val_sparse_categorical_accuracy: 0.5677
Epoch 690/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8275 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0978 - val_sparse_categorical_accuracy: 0.5677
Epoch 691/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8306 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0953 - val_sparse_categorical_accuracy: 0.5633
Epoch 692/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8251 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0764 - val_sparse_categorical_accuracy: 0.5677
Epoch 693/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8271 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5939
Epoch 694/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8262 - sparse_categorical_accuracy: 0.6569 - val_loss: 1.1247 - val_sparse_categorical_accuracy: 0.5371
Epoch 695/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8365 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.1072 - val_sparse_categorical_accuracy: 0.5546
Epoch 696/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8290 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0687 - val_sparse_categorical_accuracy: 0.5546
Epoch 697/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8285 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0704 - val_sparse_categorical_accuracy: 0.5677
Epoch 698/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8304 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1031 - val_sparse_categorical_accuracy: 0.5153
Epoch 699/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8347 - sparse_categorical_accuracy: 0.6496 - val_loss: 1.0872 - val_sparse_categorical_accuracy: 0.5764
Epoch 700/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8272 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0757 - val_sparse_categorical_accuracy: 0.5677
Epoch 701/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8267 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.1027 - val_sparse_categorical_accuracy: 0.5590
Epoch 702/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8285 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1248 - val_sparse_categorical_accuracy: 0.5459
Epoch 703/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8330 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0761 - val_sparse_categorical_accuracy: 0.5852
Epoch 704/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.8286 - sparse_categorical_accuracy: 0.6584 - val_loss: 1.0647 - val_sparse_categorical_accuracy: 0.5721
Epoch 705/2000
2/2 [==============================] - 0s 25ms/step - loss: 0.8274 - sparse_categorical_accuracy: 0.6511 - val_loss: 1.0966 - val_sparse_categorical_accuracy: 0.5677
Epoch 706/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.8260 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0952 - val_sparse_categorical_accuracy: 0.5764
Epoch 707/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.8221 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0726 - val_sparse_categorical_accuracy: 0.5721
Epoch 708/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.8244 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0756 - val_sparse_categorical_accuracy: 0.5939
Epoch 709/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8235 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0790 - val_sparse_categorical_accuracy: 0.5764
Epoch 710/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8221 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0733 - val_sparse_categorical_accuracy: 0.5852
Epoch 711/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8222 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0677 - val_sparse_categorical_accuracy: 0.5677
Epoch 712/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8206 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0843 - val_sparse_categorical_accuracy: 0.5895
Epoch 713/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8193 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0877 - val_sparse_categorical_accuracy: 0.5852
Epoch 714/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8191 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0750 - val_sparse_categorical_accuracy: 0.5764
Epoch 715/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8180 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0790 - val_sparse_categorical_accuracy: 0.5895
Epoch 716/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8187 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0870 - val_sparse_categorical_accuracy: 0.5764
Epoch 717/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8191 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0824 - val_sparse_categorical_accuracy: 0.5764
Epoch 718/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8191 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0804 - val_sparse_categorical_accuracy: 0.5721
Epoch 719/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8181 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0829 - val_sparse_categorical_accuracy: 0.5852
Epoch 720/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8182 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0812 - val_sparse_categorical_accuracy: 0.5852
Epoch 721/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.8188 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0814 - val_sparse_categorical_accuracy: 0.5764
Epoch 722/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8180 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0983 - val_sparse_categorical_accuracy: 0.5677
Epoch 723/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8198 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0996 - val_sparse_categorical_accuracy: 0.5633
Epoch 724/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8187 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0713 - val_sparse_categorical_accuracy: 0.5764
Epoch 725/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8197 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0703 - val_sparse_categorical_accuracy: 0.5764
Epoch 726/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8190 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0942 - val_sparse_categorical_accuracy: 0.5590
Epoch 727/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8194 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.1108 - val_sparse_categorical_accuracy: 0.5371
Epoch 728/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8206 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0878 - val_sparse_categorical_accuracy: 0.5808
Epoch 729/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8191 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.0823 - val_sparse_categorical_accuracy: 0.5764
Epoch 730/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8211 - sparse_categorical_accuracy: 0.6613 - val_loss: 1.0958 - val_sparse_categorical_accuracy: 0.5764
Epoch 731/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8202 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0943 - val_sparse_categorical_accuracy: 0.5808
Epoch 732/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8198 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0853 - val_sparse_categorical_accuracy: 0.5808
Epoch 733/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8168 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0817 - val_sparse_categorical_accuracy: 0.5808
Epoch 734/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8190 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0830 - val_sparse_categorical_accuracy: 0.5633
Epoch 735/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8146 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1016 - val_sparse_categorical_accuracy: 0.5677
Epoch 736/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8198 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1012 - val_sparse_categorical_accuracy: 0.5677
Epoch 737/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8191 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0819 - val_sparse_categorical_accuracy: 0.5721
Epoch 738/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8160 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0894 - val_sparse_categorical_accuracy: 0.5808
Epoch 739/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8156 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0892 - val_sparse_categorical_accuracy: 0.5764
Epoch 740/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8143 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0712 - val_sparse_categorical_accuracy: 0.5721
Epoch 741/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8158 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0676 - val_sparse_categorical_accuracy: 0.5852
Epoch 742/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8133 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0939 - val_sparse_categorical_accuracy: 0.5590
Epoch 743/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8193 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0870 - val_sparse_categorical_accuracy: 0.5721
Epoch 744/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8150 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0716 - val_sparse_categorical_accuracy: 0.5721
Epoch 745/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8191 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0838 - val_sparse_categorical_accuracy: 0.5677
Epoch 746/2000
2/2 [==============================] - 0s 20ms/step - loss: 0.8165 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0953 - val_sparse_categorical_accuracy: 0.5721
Epoch 747/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8158 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0771 - val_sparse_categorical_accuracy: 0.5808
Epoch 748/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8135 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0676 - val_sparse_categorical_accuracy: 0.5764
Epoch 749/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8152 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0744 - val_sparse_categorical_accuracy: 0.5721
Epoch 750/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8141 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0835 - val_sparse_categorical_accuracy: 0.5590
Epoch 751/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8156 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0771 - val_sparse_categorical_accuracy: 0.5808
Epoch 752/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8132 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0777 - val_sparse_categorical_accuracy: 0.5677
Epoch 753/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8161 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.1012 - val_sparse_categorical_accuracy: 0.5546
Epoch 754/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8161 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.1019 - val_sparse_categorical_accuracy: 0.5459
Epoch 755/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8137 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0890 - val_sparse_categorical_accuracy: 0.5808
Epoch 756/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8128 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0815 - val_sparse_categorical_accuracy: 0.5677
Epoch 757/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8150 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0822 - val_sparse_categorical_accuracy: 0.5633
Epoch 758/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8109 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.1012 - val_sparse_categorical_accuracy: 0.5677
Epoch 759/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8157 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0738 - val_sparse_categorical_accuracy: 0.5895
Epoch 760/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8093 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0543 - val_sparse_categorical_accuracy: 0.5546
Epoch 761/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8277 - sparse_categorical_accuracy: 0.6409 - val_loss: 1.0640 - val_sparse_categorical_accuracy: 0.5852
Epoch 762/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8099 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.1217 - val_sparse_categorical_accuracy: 0.5197
Epoch 763/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8228 - sparse_categorical_accuracy: 0.6555 - val_loss: 1.1068 - val_sparse_categorical_accuracy: 0.5459
Epoch 764/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8128 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0821 - val_sparse_categorical_accuracy: 0.5721
Epoch 765/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8204 - sparse_categorical_accuracy: 0.6657 - val_loss: 1.0800 - val_sparse_categorical_accuracy: 0.5721
Epoch 766/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.8133 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.1018 - val_sparse_categorical_accuracy: 0.5459
Epoch 767/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8203 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0736 - val_sparse_categorical_accuracy: 0.5764
Epoch 768/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8094 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0674 - val_sparse_categorical_accuracy: 0.5502
Epoch 769/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8140 - sparse_categorical_accuracy: 0.6599 - val_loss: 1.0760 - val_sparse_categorical_accuracy: 0.5677
Epoch 770/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8135 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0875 - val_sparse_categorical_accuracy: 0.5721
Epoch 771/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8130 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0894 - val_sparse_categorical_accuracy: 0.5677
Epoch 772/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8107 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0768 - val_sparse_categorical_accuracy: 0.5895
Epoch 773/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8084 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0817 - val_sparse_categorical_accuracy: 0.5808
Epoch 774/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8111 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0897 - val_sparse_categorical_accuracy: 0.5764
Epoch 775/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8105 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0998 - val_sparse_categorical_accuracy: 0.5677
Epoch 776/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8111 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0929 - val_sparse_categorical_accuracy: 0.5677
Epoch 777/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8109 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0737 - val_sparse_categorical_accuracy: 0.5808
Epoch 778/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8077 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0766 - val_sparse_categorical_accuracy: 0.5808
Epoch 779/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8103 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0750 - val_sparse_categorical_accuracy: 0.5808
Epoch 780/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8087 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0824 - val_sparse_categorical_accuracy: 0.5721
Epoch 781/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8107 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0927 - val_sparse_categorical_accuracy: 0.5721
Epoch 782/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8088 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0950 - val_sparse_categorical_accuracy: 0.5546
Epoch 783/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8101 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0864 - val_sparse_categorical_accuracy: 0.5721
Epoch 784/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8070 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0749 - val_sparse_categorical_accuracy: 0.5939
Epoch 785/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8094 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0807 - val_sparse_categorical_accuracy: 0.5808
Epoch 786/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8098 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0946 - val_sparse_categorical_accuracy: 0.5677
Epoch 787/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8080 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0873 - val_sparse_categorical_accuracy: 0.5764
Epoch 788/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8058 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0909 - val_sparse_categorical_accuracy: 0.5764
Epoch 789/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8076 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0871 - val_sparse_categorical_accuracy: 0.5764
Epoch 790/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8061 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0754 - val_sparse_categorical_accuracy: 0.5895
Epoch 791/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8059 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0731 - val_sparse_categorical_accuracy: 0.5808
Epoch 792/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8064 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0782 - val_sparse_categorical_accuracy: 0.5808
Epoch 793/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8072 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0845 - val_sparse_categorical_accuracy: 0.5721
Epoch 794/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8057 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0738 - val_sparse_categorical_accuracy: 0.5808
Epoch 795/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8064 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0773 - val_sparse_categorical_accuracy: 0.5852
Epoch 796/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8075 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0807 - val_sparse_categorical_accuracy: 0.5808
Epoch 797/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8061 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0900 - val_sparse_categorical_accuracy: 0.5633
Epoch 798/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8095 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0898 - val_sparse_categorical_accuracy: 0.5590
Epoch 799/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8065 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0723 - val_sparse_categorical_accuracy: 0.5721
Epoch 800/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8088 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0750 - val_sparse_categorical_accuracy: 0.5721
Epoch 801/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8081 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0980 - val_sparse_categorical_accuracy: 0.5764
Epoch 802/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8063 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1135 - val_sparse_categorical_accuracy: 0.5590
Epoch 803/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8094 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0997 - val_sparse_categorical_accuracy: 0.5895
Epoch 804/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8074 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0818 - val_sparse_categorical_accuracy: 0.5895
Epoch 805/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8055 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0898 - val_sparse_categorical_accuracy: 0.5764
Epoch 806/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8066 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.1115 - val_sparse_categorical_accuracy: 0.5590
Epoch 807/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8071 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1021 - val_sparse_categorical_accuracy: 0.5633
Epoch 808/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8084 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0877 - val_sparse_categorical_accuracy: 0.5808
Epoch 809/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8046 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0820 - val_sparse_categorical_accuracy: 0.5852
Epoch 810/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8035 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0804 - val_sparse_categorical_accuracy: 0.5764
Epoch 811/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.8065 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0772 - val_sparse_categorical_accuracy: 0.5895
Epoch 812/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8058 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0846 - val_sparse_categorical_accuracy: 0.5808
Epoch 813/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8065 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0967 - val_sparse_categorical_accuracy: 0.5633
Epoch 814/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8031 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0780 - val_sparse_categorical_accuracy: 0.5764
Epoch 815/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8076 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0761 - val_sparse_categorical_accuracy: 0.5764
Epoch 816/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8084 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0950 - val_sparse_categorical_accuracy: 0.5502
Epoch 817/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8016 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0916 - val_sparse_categorical_accuracy: 0.5808
Epoch 818/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8051 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0865 - val_sparse_categorical_accuracy: 0.5677
Epoch 819/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8035 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0909 - val_sparse_categorical_accuracy: 0.5721
Epoch 820/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8024 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0863 - val_sparse_categorical_accuracy: 0.5895
Epoch 821/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8026 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0757 - val_sparse_categorical_accuracy: 0.5764
Epoch 822/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8053 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0819 - val_sparse_categorical_accuracy: 0.5852
Epoch 823/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8005 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1160 - val_sparse_categorical_accuracy: 0.5197
Epoch 824/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8083 - sparse_categorical_accuracy: 0.6628 - val_loss: 1.1053 - val_sparse_categorical_accuracy: 0.5459
Epoch 825/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8039 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0762 - val_sparse_categorical_accuracy: 0.5852
Epoch 826/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8043 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0803 - val_sparse_categorical_accuracy: 0.5721
Epoch 827/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8020 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1044 - val_sparse_categorical_accuracy: 0.5546
Epoch 828/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8042 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0960 - val_sparse_categorical_accuracy: 0.5677
Epoch 829/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8031 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0927 - val_sparse_categorical_accuracy: 0.5721
Epoch 830/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8014 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0938 - val_sparse_categorical_accuracy: 0.5677
Epoch 831/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7999 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0809 - val_sparse_categorical_accuracy: 0.5939
Epoch 832/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8001 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0792 - val_sparse_categorical_accuracy: 0.5808
Epoch 833/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8015 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0874 - val_sparse_categorical_accuracy: 0.5764
Epoch 834/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7996 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0964 - val_sparse_categorical_accuracy: 0.5677
Epoch 835/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8009 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0863 - val_sparse_categorical_accuracy: 0.5852
Epoch 836/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7978 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0781 - val_sparse_categorical_accuracy: 0.5939
Epoch 837/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8036 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0827 - val_sparse_categorical_accuracy: 0.5677
Epoch 838/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7987 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1091 - val_sparse_categorical_accuracy: 0.5328
Epoch 839/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8073 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0974 - val_sparse_categorical_accuracy: 0.5502
Epoch 840/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8005 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0780 - val_sparse_categorical_accuracy: 0.5590
Epoch 841/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8040 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0856 - val_sparse_categorical_accuracy: 0.5633
Epoch 842/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7989 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1022 - val_sparse_categorical_accuracy: 0.5590
Epoch 843/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8003 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0965 - val_sparse_categorical_accuracy: 0.5633
Epoch 844/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7982 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0816 - val_sparse_categorical_accuracy: 0.5808
Epoch 845/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7988 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0821 - val_sparse_categorical_accuracy: 0.5808
Epoch 846/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7977 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0941 - val_sparse_categorical_accuracy: 0.5633
Epoch 847/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7983 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1026 - val_sparse_categorical_accuracy: 0.5415
Epoch 848/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7982 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0850 - val_sparse_categorical_accuracy: 0.5808
Epoch 849/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7966 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0775 - val_sparse_categorical_accuracy: 0.5764
Epoch 850/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8022 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0900 - val_sparse_categorical_accuracy: 0.5852
Epoch 851/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7977 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1017 - val_sparse_categorical_accuracy: 0.5677
Epoch 852/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7983 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0907 - val_sparse_categorical_accuracy: 0.5677
Epoch 853/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7984 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0840 - val_sparse_categorical_accuracy: 0.5852
Epoch 854/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7968 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0951 - val_sparse_categorical_accuracy: 0.5633
Epoch 855/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7967 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.0947 - val_sparse_categorical_accuracy: 0.5677
Epoch 856/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7962 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0920 - val_sparse_categorical_accuracy: 0.5764
Epoch 857/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7977 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0865 - val_sparse_categorical_accuracy: 0.5852
Epoch 858/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7980 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0801 - val_sparse_categorical_accuracy: 0.5852
Epoch 859/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7964 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0942 - val_sparse_categorical_accuracy: 0.5721
Epoch 860/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7971 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0932 - val_sparse_categorical_accuracy: 0.5677
Epoch 861/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7955 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0846 - val_sparse_categorical_accuracy: 0.5677
Epoch 862/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7986 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0835 - val_sparse_categorical_accuracy: 0.5590
Epoch 863/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8002 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0930 - val_sparse_categorical_accuracy: 0.5677
Epoch 864/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7955 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1011 - val_sparse_categorical_accuracy: 0.5633
Epoch 865/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7957 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0862 - val_sparse_categorical_accuracy: 0.5852
Epoch 866/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7960 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0766 - val_sparse_categorical_accuracy: 0.5983
Epoch 867/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7960 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0952 - val_sparse_categorical_accuracy: 0.5415
Epoch 868/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7973 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0913 - val_sparse_categorical_accuracy: 0.5852
Epoch 869/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7961 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0757 - val_sparse_categorical_accuracy: 0.5721
Epoch 870/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7978 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0906 - val_sparse_categorical_accuracy: 0.5546
Epoch 871/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7989 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1050 - val_sparse_categorical_accuracy: 0.5546
Epoch 872/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7992 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0721 - val_sparse_categorical_accuracy: 0.5808
Epoch 873/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8007 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0731 - val_sparse_categorical_accuracy: 0.5808
Epoch 874/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7939 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1103 - val_sparse_categorical_accuracy: 0.5371
Epoch 875/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.8010 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.1168 - val_sparse_categorical_accuracy: 0.5240
Epoch 876/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7978 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0817 - val_sparse_categorical_accuracy: 0.5808
Epoch 877/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7969 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0673 - val_sparse_categorical_accuracy: 0.5808
Epoch 878/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7972 - sparse_categorical_accuracy: 0.6672 - val_loss: 1.0778 - val_sparse_categorical_accuracy: 0.5983
Epoch 879/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7992 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0770 - val_sparse_categorical_accuracy: 0.5852
Epoch 880/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7947 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0734 - val_sparse_categorical_accuracy: 0.5677
Epoch 881/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7995 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0800 - val_sparse_categorical_accuracy: 0.5895
Epoch 882/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7973 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0886 - val_sparse_categorical_accuracy: 0.5677
Epoch 883/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7948 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0812 - val_sparse_categorical_accuracy: 0.5764
Epoch 884/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7928 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0951 - val_sparse_categorical_accuracy: 0.5764
Epoch 885/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7971 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.1003 - val_sparse_categorical_accuracy: 0.5633
Epoch 886/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7933 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.1052 - val_sparse_categorical_accuracy: 0.5502
Epoch 887/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7981 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0751 - val_sparse_categorical_accuracy: 0.5808
Epoch 888/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7949 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0694 - val_sparse_categorical_accuracy: 0.5852
Epoch 889/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.8020 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0854 - val_sparse_categorical_accuracy: 0.5895
Epoch 890/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7955 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0985 - val_sparse_categorical_accuracy: 0.5633
Epoch 891/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7943 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0988 - val_sparse_categorical_accuracy: 0.5764
Epoch 892/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7968 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0941 - val_sparse_categorical_accuracy: 0.5721
Epoch 893/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7933 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0908 - val_sparse_categorical_accuracy: 0.5677
Epoch 894/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7942 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0995 - val_sparse_categorical_accuracy: 0.5459
Epoch 895/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7970 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0846 - val_sparse_categorical_accuracy: 0.5677
Epoch 896/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7987 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0837 - val_sparse_categorical_accuracy: 0.5852
Epoch 897/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7942 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0955 - val_sparse_categorical_accuracy: 0.5590
Epoch 898/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7947 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0974 - val_sparse_categorical_accuracy: 0.5590
Epoch 899/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7936 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0865 - val_sparse_categorical_accuracy: 0.5895
Epoch 900/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7936 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0852 - val_sparse_categorical_accuracy: 0.5852
Epoch 901/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7974 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0874 - val_sparse_categorical_accuracy: 0.5677
Epoch 902/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7903 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0731 - val_sparse_categorical_accuracy: 0.5764
Epoch 903/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7951 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0804 - val_sparse_categorical_accuracy: 0.5808
Epoch 904/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.7904 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1064 - val_sparse_categorical_accuracy: 0.5590
Epoch 905/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7919 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1010 - val_sparse_categorical_accuracy: 0.5677
Epoch 906/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7900 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0854 - val_sparse_categorical_accuracy: 0.5983
Epoch 907/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7919 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0769 - val_sparse_categorical_accuracy: 0.5939
Epoch 908/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7900 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0856 - val_sparse_categorical_accuracy: 0.5721
Epoch 909/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7919 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0911 - val_sparse_categorical_accuracy: 0.5502
Epoch 910/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7911 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0883 - val_sparse_categorical_accuracy: 0.5677
Epoch 911/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7920 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0841 - val_sparse_categorical_accuracy: 0.5808
Epoch 912/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7906 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0815 - val_sparse_categorical_accuracy: 0.5895
Epoch 913/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7895 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5764
Epoch 914/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7928 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5721
Epoch 915/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7914 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0932 - val_sparse_categorical_accuracy: 0.5808
Epoch 916/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7891 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0939 - val_sparse_categorical_accuracy: 0.5721
Epoch 917/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7901 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1002 - val_sparse_categorical_accuracy: 0.5633
Epoch 918/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7922 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0942 - val_sparse_categorical_accuracy: 0.5808
Epoch 919/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7882 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0816 - val_sparse_categorical_accuracy: 0.5983
Epoch 920/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7917 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0926 - val_sparse_categorical_accuracy: 0.5677
Epoch 921/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7908 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1056 - val_sparse_categorical_accuracy: 0.5328
Epoch 922/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7920 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0826 - val_sparse_categorical_accuracy: 0.5808
Epoch 923/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7920 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0759 - val_sparse_categorical_accuracy: 0.5764
Epoch 924/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7909 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0978 - val_sparse_categorical_accuracy: 0.5459
Epoch 925/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7938 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1035 - val_sparse_categorical_accuracy: 0.5415
Epoch 926/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7892 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0883 - val_sparse_categorical_accuracy: 0.5983
Epoch 927/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7937 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0904 - val_sparse_categorical_accuracy: 0.5852
Epoch 928/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7855 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1090 - val_sparse_categorical_accuracy: 0.5459
Epoch 929/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7932 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0945 - val_sparse_categorical_accuracy: 0.5590
Epoch 930/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7893 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0700 - val_sparse_categorical_accuracy: 0.5764
Epoch 931/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7920 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0795 - val_sparse_categorical_accuracy: 0.5808
Epoch 932/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7929 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.1028 - val_sparse_categorical_accuracy: 0.5502
Epoch 933/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7909 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1076 - val_sparse_categorical_accuracy: 0.5415
Epoch 934/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7899 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0802 - val_sparse_categorical_accuracy: 0.5633
Epoch 935/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7932 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0763 - val_sparse_categorical_accuracy: 0.5721
Epoch 936/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7900 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0989 - val_sparse_categorical_accuracy: 0.5546
Epoch 937/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7896 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1034 - val_sparse_categorical_accuracy: 0.5502
Epoch 938/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7886 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0868 - val_sparse_categorical_accuracy: 0.5895
Epoch 939/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7931 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0887 - val_sparse_categorical_accuracy: 0.5764
Epoch 940/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7877 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.1160 - val_sparse_categorical_accuracy: 0.5197
Epoch 941/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7953 - sparse_categorical_accuracy: 0.6642 - val_loss: 1.0900 - val_sparse_categorical_accuracy: 0.5764
Epoch 942/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7863 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0840 - val_sparse_categorical_accuracy: 0.5764
Epoch 943/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7856 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1014 - val_sparse_categorical_accuracy: 0.5546
Epoch 944/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7861 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1083 - val_sparse_categorical_accuracy: 0.5502
Epoch 945/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7858 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0944 - val_sparse_categorical_accuracy: 0.5852
Epoch 946/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7879 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0921 - val_sparse_categorical_accuracy: 0.5808
Epoch 947/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7905 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0940 - val_sparse_categorical_accuracy: 0.5808
Epoch 948/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7851 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0851 - val_sparse_categorical_accuracy: 0.5721
Epoch 949/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7857 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0889 - val_sparse_categorical_accuracy: 0.5633
Epoch 950/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7878 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0913 - val_sparse_categorical_accuracy: 0.5808
Epoch 951/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7891 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0975 - val_sparse_categorical_accuracy: 0.5852
Epoch 952/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7842 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0973 - val_sparse_categorical_accuracy: 0.5721
Epoch 953/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7859 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.0874 - val_sparse_categorical_accuracy: 0.5808
Epoch 954/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7858 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0877 - val_sparse_categorical_accuracy: 0.5852
Epoch 955/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7867 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1015 - val_sparse_categorical_accuracy: 0.5546
Epoch 956/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7845 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1030 - val_sparse_categorical_accuracy: 0.5677
Epoch 957/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7864 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0946 - val_sparse_categorical_accuracy: 0.5939
Epoch 958/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7815 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1018 - val_sparse_categorical_accuracy: 0.5677
Epoch 959/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7848 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1039 - val_sparse_categorical_accuracy: 0.5415
Epoch 960/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7865 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0967 - val_sparse_categorical_accuracy: 0.5633
Epoch 961/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7832 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0917 - val_sparse_categorical_accuracy: 0.5721
Epoch 962/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7880 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0884 - val_sparse_categorical_accuracy: 0.5895
Epoch 963/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7824 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0973 - val_sparse_categorical_accuracy: 0.5677
Epoch 964/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7887 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0820 - val_sparse_categorical_accuracy: 0.5721
Epoch 965/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7854 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0683 - val_sparse_categorical_accuracy: 0.5764
Epoch 966/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7847 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0882 - val_sparse_categorical_accuracy: 0.5721
Epoch 967/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7854 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1081 - val_sparse_categorical_accuracy: 0.5677
Epoch 968/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7820 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.1027 - val_sparse_categorical_accuracy: 0.5808
Epoch 969/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7827 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0877 - val_sparse_categorical_accuracy: 0.5852
Epoch 970/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7837 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0789 - val_sparse_categorical_accuracy: 0.5852
Epoch 971/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7849 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0928 - val_sparse_categorical_accuracy: 0.5677
Epoch 972/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7859 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1079 - val_sparse_categorical_accuracy: 0.5284
Epoch 973/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7856 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0678 - val_sparse_categorical_accuracy: 0.5808
Epoch 974/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7850 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0694 - val_sparse_categorical_accuracy: 0.5764
Epoch 975/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7869 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0984 - val_sparse_categorical_accuracy: 0.5546
Epoch 976/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7824 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1106 - val_sparse_categorical_accuracy: 0.5546
Epoch 977/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7868 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0796 - val_sparse_categorical_accuracy: 0.5677
Epoch 978/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7887 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0774 - val_sparse_categorical_accuracy: 0.5895
Epoch 979/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7855 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1057 - val_sparse_categorical_accuracy: 0.5677
Epoch 980/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7847 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0935 - val_sparse_categorical_accuracy: 0.5939
Epoch 981/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7786 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0821 - val_sparse_categorical_accuracy: 0.5808
Epoch 982/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7822 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0856 - val_sparse_categorical_accuracy: 0.5721
Epoch 983/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7810 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1039 - val_sparse_categorical_accuracy: 0.5197
Epoch 984/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7901 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5546
Epoch 985/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7838 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0805 - val_sparse_categorical_accuracy: 0.5808
Epoch 986/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7853 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0888 - val_sparse_categorical_accuracy: 0.5721
Epoch 987/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7797 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0935 - val_sparse_categorical_accuracy: 0.5546
Epoch 988/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7799 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0772 - val_sparse_categorical_accuracy: 0.5852
Epoch 989/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7816 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0811 - val_sparse_categorical_accuracy: 0.5852
Epoch 990/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7789 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1019 - val_sparse_categorical_accuracy: 0.5546
Epoch 991/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7798 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0938 - val_sparse_categorical_accuracy: 0.5721
Epoch 992/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7769 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0740 - val_sparse_categorical_accuracy: 0.5764
Epoch 993/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7802 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0691 - val_sparse_categorical_accuracy: 0.5808
Epoch 994/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7786 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0890 - val_sparse_categorical_accuracy: 0.5633
Epoch 995/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7811 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0936 - val_sparse_categorical_accuracy: 0.5590
Epoch 996/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7795 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0851 - val_sparse_categorical_accuracy: 0.5852
Epoch 997/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7820 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0890 - val_sparse_categorical_accuracy: 0.5895
Epoch 998/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7768 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0941 - val_sparse_categorical_accuracy: 0.5633
Epoch 999/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7779 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0891 - val_sparse_categorical_accuracy: 0.5721
Epoch 1000/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7778 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0867 - val_sparse_categorical_accuracy: 0.5721
Epoch 1001/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7756 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0960 - val_sparse_categorical_accuracy: 0.5633
Epoch 1002/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7765 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1054 - val_sparse_categorical_accuracy: 0.5677
Epoch 1003/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7769 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0938 - val_sparse_categorical_accuracy: 0.5852
Epoch 1004/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7767 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0771 - val_sparse_categorical_accuracy: 0.5721
Epoch 1005/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7792 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0758 - val_sparse_categorical_accuracy: 0.5721
Epoch 1006/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7822 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0808 - val_sparse_categorical_accuracy: 0.5721
Epoch 1007/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7764 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0772 - val_sparse_categorical_accuracy: 0.5721
Epoch 1008/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7813 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0912 - val_sparse_categorical_accuracy: 0.5852
Epoch 1009/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7782 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1043 - val_sparse_categorical_accuracy: 0.5590
Epoch 1010/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7793 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0803 - val_sparse_categorical_accuracy: 0.5764
Epoch 1011/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7782 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0772 - val_sparse_categorical_accuracy: 0.5764
Epoch 1012/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.7799 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1007 - val_sparse_categorical_accuracy: 0.5633
Epoch 1013/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7773 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1122 - val_sparse_categorical_accuracy: 0.5415
Epoch 1014/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7787 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0852 - val_sparse_categorical_accuracy: 0.5764
Epoch 1015/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7740 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0761 - val_sparse_categorical_accuracy: 0.5590
Epoch 1016/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7796 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0849 - val_sparse_categorical_accuracy: 0.5764
Epoch 1017/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7752 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1104 - val_sparse_categorical_accuracy: 0.5459
Epoch 1018/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7789 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1023 - val_sparse_categorical_accuracy: 0.5852
Epoch 1019/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7760 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0888 - val_sparse_categorical_accuracy: 0.5808
Epoch 1020/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7777 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0834 - val_sparse_categorical_accuracy: 0.5633
Epoch 1021/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7783 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0900 - val_sparse_categorical_accuracy: 0.5764
Epoch 1022/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7744 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0963 - val_sparse_categorical_accuracy: 0.5764
Epoch 1023/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7758 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0907 - val_sparse_categorical_accuracy: 0.5895
Epoch 1024/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7764 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0888 - val_sparse_categorical_accuracy: 0.5677
Epoch 1025/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7751 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0919 - val_sparse_categorical_accuracy: 0.5590
Epoch 1026/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7783 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0828 - val_sparse_categorical_accuracy: 0.5808
Epoch 1027/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7736 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0885 - val_sparse_categorical_accuracy: 0.5764
Epoch 1028/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7823 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0927 - val_sparse_categorical_accuracy: 0.5677
Epoch 1029/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7780 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1020 - val_sparse_categorical_accuracy: 0.5415
Epoch 1030/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7756 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0909 - val_sparse_categorical_accuracy: 0.5721
Epoch 1031/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7793 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0980 - val_sparse_categorical_accuracy: 0.5590
Epoch 1032/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7740 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1174 - val_sparse_categorical_accuracy: 0.5197
Epoch 1033/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7856 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0988 - val_sparse_categorical_accuracy: 0.5459
Epoch 1034/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7828 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0740 - val_sparse_categorical_accuracy: 0.5764
Epoch 1035/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7817 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0897 - val_sparse_categorical_accuracy: 0.5721
Epoch 1036/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7802 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0997 - val_sparse_categorical_accuracy: 0.5502
Epoch 1037/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7806 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0847 - val_sparse_categorical_accuracy: 0.5721
Epoch 1038/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7790 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0880 - val_sparse_categorical_accuracy: 0.5721
Epoch 1039/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7814 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.1008 - val_sparse_categorical_accuracy: 0.5764
Epoch 1040/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7764 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.1146 - val_sparse_categorical_accuracy: 0.5633
Epoch 1041/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7796 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.1006 - val_sparse_categorical_accuracy: 0.5677
Epoch 1042/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7780 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0869 - val_sparse_categorical_accuracy: 0.5808
Epoch 1043/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7759 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5808
Epoch 1044/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7749 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0941 - val_sparse_categorical_accuracy: 0.5677
Epoch 1045/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7731 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0807 - val_sparse_categorical_accuracy: 0.5502
Epoch 1046/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7740 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0759 - val_sparse_categorical_accuracy: 0.5633
Epoch 1047/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7735 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0871 - val_sparse_categorical_accuracy: 0.5983
Epoch 1048/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7756 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0813 - val_sparse_categorical_accuracy: 0.5764
Epoch 1049/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7729 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0735 - val_sparse_categorical_accuracy: 0.5677
Epoch 1050/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7740 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0974 - val_sparse_categorical_accuracy: 0.5677
Epoch 1051/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7771 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.1116 - val_sparse_categorical_accuracy: 0.5677
Epoch 1052/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7754 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0940 - val_sparse_categorical_accuracy: 0.5808
Epoch 1053/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7745 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0878 - val_sparse_categorical_accuracy: 0.5633
Epoch 1054/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7746 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1039 - val_sparse_categorical_accuracy: 0.5677
Epoch 1055/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7781 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1109 - val_sparse_categorical_accuracy: 0.5459
Epoch 1056/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7749 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0874 - val_sparse_categorical_accuracy: 0.5633
Epoch 1057/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7796 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0810 - val_sparse_categorical_accuracy: 0.5633
Epoch 1058/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7755 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0990 - val_sparse_categorical_accuracy: 0.5590
Epoch 1059/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7740 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0888 - val_sparse_categorical_accuracy: 0.5895
Epoch 1060/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7716 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0884 - val_sparse_categorical_accuracy: 0.5895
Epoch 1061/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7723 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0872 - val_sparse_categorical_accuracy: 0.5895
Epoch 1062/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7688 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0795 - val_sparse_categorical_accuracy: 0.5852
Epoch 1063/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7715 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0847 - val_sparse_categorical_accuracy: 0.5677
Epoch 1064/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7705 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1023 - val_sparse_categorical_accuracy: 0.5764
Epoch 1065/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7705 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1053 - val_sparse_categorical_accuracy: 0.5633
Epoch 1066/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7721 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0847 - val_sparse_categorical_accuracy: 0.5852
Epoch 1067/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7718 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0771 - val_sparse_categorical_accuracy: 0.5764
Epoch 1068/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7721 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0893 - val_sparse_categorical_accuracy: 0.5764
Epoch 1069/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7679 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1181 - val_sparse_categorical_accuracy: 0.5502
Epoch 1070/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7750 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1011 - val_sparse_categorical_accuracy: 0.5764
Epoch 1071/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7754 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0825 - val_sparse_categorical_accuracy: 0.5764
Epoch 1072/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7722 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0942 - val_sparse_categorical_accuracy: 0.5721
Epoch 1073/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7744 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0895 - val_sparse_categorical_accuracy: 0.5939
Epoch 1074/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7695 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0883 - val_sparse_categorical_accuracy: 0.5633
Epoch 1075/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7753 - sparse_categorical_accuracy: 0.6715 - val_loss: 1.0909 - val_sparse_categorical_accuracy: 0.5677
Epoch 1076/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7705 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1123 - val_sparse_categorical_accuracy: 0.5240
Epoch 1077/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7762 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0951 - val_sparse_categorical_accuracy: 0.5721
Epoch 1078/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7675 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0902 - val_sparse_categorical_accuracy: 0.5633
Epoch 1079/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7752 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0908 - val_sparse_categorical_accuracy: 0.5721
Epoch 1080/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7717 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1012 - val_sparse_categorical_accuracy: 0.5764
Epoch 1081/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7725 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1024 - val_sparse_categorical_accuracy: 0.5895
Epoch 1082/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7679 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0943 - val_sparse_categorical_accuracy: 0.5721
Epoch 1083/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7726 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0985 - val_sparse_categorical_accuracy: 0.5764
Epoch 1084/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7666 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1186 - val_sparse_categorical_accuracy: 0.5328
Epoch 1085/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7748 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0955 - val_sparse_categorical_accuracy: 0.5590
Epoch 1086/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7715 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0796 - val_sparse_categorical_accuracy: 0.5459
Epoch 1087/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7714 - sparse_categorical_accuracy: 0.6686 - val_loss: 1.0919 - val_sparse_categorical_accuracy: 0.5633
Epoch 1088/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7673 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1183 - val_sparse_categorical_accuracy: 0.5284
Epoch 1089/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7760 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0873 - val_sparse_categorical_accuracy: 0.5764
Epoch 1090/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7668 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0808 - val_sparse_categorical_accuracy: 0.5546
Epoch 1091/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.7792 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0897 - val_sparse_categorical_accuracy: 0.5764
Epoch 1092/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7653 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1276 - val_sparse_categorical_accuracy: 0.5197
Epoch 1093/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7779 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1015 - val_sparse_categorical_accuracy: 0.5721
Epoch 1094/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7671 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0822 - val_sparse_categorical_accuracy: 0.5546
Epoch 1095/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7691 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.0875 - val_sparse_categorical_accuracy: 0.5721
Epoch 1096/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7660 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1069 - val_sparse_categorical_accuracy: 0.5459
Epoch 1097/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7708 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5764
Epoch 1098/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7689 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0772 - val_sparse_categorical_accuracy: 0.5633
Epoch 1099/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7678 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0908 - val_sparse_categorical_accuracy: 0.5764
Epoch 1100/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7663 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.0981 - val_sparse_categorical_accuracy: 0.5852
Epoch 1101/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7654 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0917 - val_sparse_categorical_accuracy: 0.5808
Epoch 1102/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7679 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0845 - val_sparse_categorical_accuracy: 0.5808
Epoch 1103/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7649 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0957 - val_sparse_categorical_accuracy: 0.5721
Epoch 1104/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7666 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0966 - val_sparse_categorical_accuracy: 0.5721
Epoch 1105/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7648 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0848 - val_sparse_categorical_accuracy: 0.5677
Epoch 1106/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7668 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0793 - val_sparse_categorical_accuracy: 0.5808
Epoch 1107/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7666 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0960 - val_sparse_categorical_accuracy: 0.5677
Epoch 1108/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7638 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1009 - val_sparse_categorical_accuracy: 0.5939
Epoch 1109/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7664 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0959 - val_sparse_categorical_accuracy: 0.5939
Epoch 1110/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7654 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0961 - val_sparse_categorical_accuracy: 0.5633
Epoch 1111/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7656 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0941 - val_sparse_categorical_accuracy: 0.5677
Epoch 1112/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.7651 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0838 - val_sparse_categorical_accuracy: 0.5677
Epoch 1113/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7664 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0845 - val_sparse_categorical_accuracy: 0.5633
Epoch 1114/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7657 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0995 - val_sparse_categorical_accuracy: 0.5590
Epoch 1115/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7647 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0959 - val_sparse_categorical_accuracy: 0.5939
Epoch 1116/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7627 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0896 - val_sparse_categorical_accuracy: 0.5546
Epoch 1117/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7688 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0881 - val_sparse_categorical_accuracy: 0.5677
Epoch 1118/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7624 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0979 - val_sparse_categorical_accuracy: 0.5415
Epoch 1119/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7658 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0863 - val_sparse_categorical_accuracy: 0.5633
Epoch 1120/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7628 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0900 - val_sparse_categorical_accuracy: 0.5677
Epoch 1121/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7698 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0993 - val_sparse_categorical_accuracy: 0.5721
Epoch 1122/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7619 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1097 - val_sparse_categorical_accuracy: 0.5590
Epoch 1123/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7707 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0892 - val_sparse_categorical_accuracy: 0.5721
Epoch 1124/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7674 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0842 - val_sparse_categorical_accuracy: 0.5852
Epoch 1125/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7666 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0980 - val_sparse_categorical_accuracy: 0.5852
Epoch 1126/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7678 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0990 - val_sparse_categorical_accuracy: 0.5502
Epoch 1127/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7676 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0824 - val_sparse_categorical_accuracy: 0.5677
Epoch 1128/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7663 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0795 - val_sparse_categorical_accuracy: 0.5677
Epoch 1129/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7665 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0845 - val_sparse_categorical_accuracy: 0.5721
Epoch 1130/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7628 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0875 - val_sparse_categorical_accuracy: 0.5808
Epoch 1131/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7655 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0858 - val_sparse_categorical_accuracy: 0.5764
Epoch 1132/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7619 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0895 - val_sparse_categorical_accuracy: 0.5852
Epoch 1133/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7639 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1005 - val_sparse_categorical_accuracy: 0.5677
Epoch 1134/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7622 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1075 - val_sparse_categorical_accuracy: 0.5546
Epoch 1135/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7665 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0876 - val_sparse_categorical_accuracy: 0.5590
Epoch 1136/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7674 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0788 - val_sparse_categorical_accuracy: 0.5677
Epoch 1137/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7649 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0955 - val_sparse_categorical_accuracy: 0.5328
Epoch 1138/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7703 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1032 - val_sparse_categorical_accuracy: 0.5459
Epoch 1139/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7691 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0917 - val_sparse_categorical_accuracy: 0.5808
Epoch 1140/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7651 - sparse_categorical_accuracy: 0.6745 - val_loss: 1.1000 - val_sparse_categorical_accuracy: 0.5633
Epoch 1141/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7662 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.1081 - val_sparse_categorical_accuracy: 0.5590
Epoch 1142/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7638 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0951 - val_sparse_categorical_accuracy: 0.5721
Epoch 1143/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7650 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0797 - val_sparse_categorical_accuracy: 0.5764
Epoch 1144/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7641 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0896 - val_sparse_categorical_accuracy: 0.5895
Epoch 1145/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7621 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1029 - val_sparse_categorical_accuracy: 0.5764
Epoch 1146/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7617 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0959 - val_sparse_categorical_accuracy: 0.5677
Epoch 1147/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7598 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0904 - val_sparse_categorical_accuracy: 0.5764
Epoch 1148/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7623 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0903 - val_sparse_categorical_accuracy: 0.5721
Epoch 1149/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7635 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0889 - val_sparse_categorical_accuracy: 0.5677
Epoch 1150/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7602 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0939 - val_sparse_categorical_accuracy: 0.5677
Epoch 1151/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7622 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0941 - val_sparse_categorical_accuracy: 0.5939
Epoch 1152/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7589 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0984 - val_sparse_categorical_accuracy: 0.5895
Epoch 1153/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7650 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0906 - val_sparse_categorical_accuracy: 0.6026
Epoch 1154/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7641 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0869 - val_sparse_categorical_accuracy: 0.5677
Epoch 1155/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7614 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.1081 - val_sparse_categorical_accuracy: 0.5764
Epoch 1156/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7673 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1119 - val_sparse_categorical_accuracy: 0.5764
Epoch 1157/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7653 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0867 - val_sparse_categorical_accuracy: 0.5852
Epoch 1158/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7612 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5852
Epoch 1159/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7644 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0907 - val_sparse_categorical_accuracy: 0.5590
Epoch 1160/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7636 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0895 - val_sparse_categorical_accuracy: 0.5677
Epoch 1161/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7625 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0921 - val_sparse_categorical_accuracy: 0.5590
Epoch 1162/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7596 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1046 - val_sparse_categorical_accuracy: 0.5721
Epoch 1163/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7613 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1033 - val_sparse_categorical_accuracy: 0.5808
Epoch 1164/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7602 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0863 - val_sparse_categorical_accuracy: 0.5852
Epoch 1165/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7595 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0653 - val_sparse_categorical_accuracy: 0.5590
Epoch 1166/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7643 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0695 - val_sparse_categorical_accuracy: 0.5633
Epoch 1167/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7639 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0914 - val_sparse_categorical_accuracy: 0.5502
Epoch 1168/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7604 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1020 - val_sparse_categorical_accuracy: 0.5633
Epoch 1169/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7619 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0987 - val_sparse_categorical_accuracy: 0.5808
Epoch 1170/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7622 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0911 - val_sparse_categorical_accuracy: 0.5721
Epoch 1171/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7604 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1010 - val_sparse_categorical_accuracy: 0.5764
Epoch 1172/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7611 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1002 - val_sparse_categorical_accuracy: 0.5895
Epoch 1173/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7599 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0944 - val_sparse_categorical_accuracy: 0.5895
Epoch 1174/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7591 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0905 - val_sparse_categorical_accuracy: 0.5808
Epoch 1175/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7581 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0911 - val_sparse_categorical_accuracy: 0.5590
Epoch 1176/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7625 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0730 - val_sparse_categorical_accuracy: 0.5677
Epoch 1177/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7612 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0768 - val_sparse_categorical_accuracy: 0.5590
Epoch 1178/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7677 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0988 - val_sparse_categorical_accuracy: 0.5633
Epoch 1179/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7631 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1239 - val_sparse_categorical_accuracy: 0.5328
Epoch 1180/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7689 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0850 - val_sparse_categorical_accuracy: 0.5721
Epoch 1181/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7620 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0881 - val_sparse_categorical_accuracy: 0.5502
Epoch 1182/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7637 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1101 - val_sparse_categorical_accuracy: 0.5764
Epoch 1183/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7591 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1151 - val_sparse_categorical_accuracy: 0.5415
Epoch 1184/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7632 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0770 - val_sparse_categorical_accuracy: 0.5852
Epoch 1185/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7596 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0683 - val_sparse_categorical_accuracy: 0.5590
Epoch 1186/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7640 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0944 - val_sparse_categorical_accuracy: 0.5633
Epoch 1187/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7593 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1349 - val_sparse_categorical_accuracy: 0.5284
Epoch 1188/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7690 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1005 - val_sparse_categorical_accuracy: 0.5895
Epoch 1189/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7595 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0788 - val_sparse_categorical_accuracy: 0.5852
Epoch 1190/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7576 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0871 - val_sparse_categorical_accuracy: 0.5633
Epoch 1191/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7603 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1059 - val_sparse_categorical_accuracy: 0.5502
Epoch 1192/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7613 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0916 - val_sparse_categorical_accuracy: 0.5721
Epoch 1193/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7567 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0882 - val_sparse_categorical_accuracy: 0.5677
Epoch 1194/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7603 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0981 - val_sparse_categorical_accuracy: 0.5459
Epoch 1195/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7618 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1063 - val_sparse_categorical_accuracy: 0.5153
Epoch 1196/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7639 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0715 - val_sparse_categorical_accuracy: 0.5677
Epoch 1197/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7616 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0700 - val_sparse_categorical_accuracy: 0.5633
Epoch 1198/2000
2/2 [==============================] - 0s 22ms/step - loss: 0.7628 - sparse_categorical_accuracy: 0.6730 - val_loss: 1.0936 - val_sparse_categorical_accuracy: 0.5895
Epoch 1199/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7623 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1089 - val_sparse_categorical_accuracy: 0.5852
Epoch 1200/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7617 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.0915 - val_sparse_categorical_accuracy: 0.5721
Epoch 1201/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7584 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0923 - val_sparse_categorical_accuracy: 0.5590
Epoch 1202/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7608 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0993 - val_sparse_categorical_accuracy: 0.5677
Epoch 1203/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7567 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0890 - val_sparse_categorical_accuracy: 0.5808
Epoch 1204/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7557 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0868 - val_sparse_categorical_accuracy: 0.5852
Epoch 1205/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7567 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0954 - val_sparse_categorical_accuracy: 0.5852
Epoch 1206/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7553 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1039 - val_sparse_categorical_accuracy: 0.5590
Epoch 1207/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7613 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0864 - val_sparse_categorical_accuracy: 0.5852
Epoch 1208/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7544 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0849 - val_sparse_categorical_accuracy: 0.5808
Epoch 1209/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7565 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0887 - val_sparse_categorical_accuracy: 0.5764
Epoch 1210/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7557 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0930 - val_sparse_categorical_accuracy: 0.5895
Epoch 1211/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7540 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0893 - val_sparse_categorical_accuracy: 0.5808
Epoch 1212/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7552 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0939 - val_sparse_categorical_accuracy: 0.5852
Epoch 1213/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7569 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0998 - val_sparse_categorical_accuracy: 0.5633
Epoch 1214/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7557 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0934 - val_sparse_categorical_accuracy: 0.5502
Epoch 1215/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7579 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0790 - val_sparse_categorical_accuracy: 0.5677
Epoch 1216/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7597 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0888 - val_sparse_categorical_accuracy: 0.5721
Epoch 1217/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7562 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1137 - val_sparse_categorical_accuracy: 0.5677
Epoch 1218/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7605 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0986 - val_sparse_categorical_accuracy: 0.5721
Epoch 1219/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7529 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0814 - val_sparse_categorical_accuracy: 0.5895
Epoch 1220/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7611 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0812 - val_sparse_categorical_accuracy: 0.5895
Epoch 1221/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7596 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0955 - val_sparse_categorical_accuracy: 0.5677
Epoch 1222/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7559 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1120 - val_sparse_categorical_accuracy: 0.5721
Epoch 1223/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7554 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1082 - val_sparse_categorical_accuracy: 0.5590
Epoch 1224/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7548 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0942 - val_sparse_categorical_accuracy: 0.5764
Epoch 1225/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7576 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0803 - val_sparse_categorical_accuracy: 0.5677
Epoch 1226/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7557 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0805 - val_sparse_categorical_accuracy: 0.5677
Epoch 1227/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7544 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0898 - val_sparse_categorical_accuracy: 0.5677
Epoch 1228/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7546 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1061 - val_sparse_categorical_accuracy: 0.5852
Epoch 1229/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7578 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0999 - val_sparse_categorical_accuracy: 0.5677
Epoch 1230/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7568 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0780 - val_sparse_categorical_accuracy: 0.5808
Epoch 1231/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7542 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0814 - val_sparse_categorical_accuracy: 0.5633
Epoch 1232/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7580 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0950 - val_sparse_categorical_accuracy: 0.5852
Epoch 1233/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7557 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1096 - val_sparse_categorical_accuracy: 0.5677
Epoch 1234/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7573 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1006 - val_sparse_categorical_accuracy: 0.5721
Epoch 1235/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7550 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0840 - val_sparse_categorical_accuracy: 0.5590
Epoch 1236/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7528 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0692 - val_sparse_categorical_accuracy: 0.5721
Epoch 1237/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7544 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0777 - val_sparse_categorical_accuracy: 0.5677
Epoch 1238/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7524 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0985 - val_sparse_categorical_accuracy: 0.5721
Epoch 1239/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7542 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1033 - val_sparse_categorical_accuracy: 0.5677
Epoch 1240/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7553 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0909 - val_sparse_categorical_accuracy: 0.5721
Epoch 1241/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7516 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0808 - val_sparse_categorical_accuracy: 0.5764
Epoch 1242/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7545 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0804 - val_sparse_categorical_accuracy: 0.5633
Epoch 1243/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7515 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1018 - val_sparse_categorical_accuracy: 0.5852
Epoch 1244/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7518 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1038 - val_sparse_categorical_accuracy: 0.5808
Epoch 1245/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7541 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0922 - val_sparse_categorical_accuracy: 0.5939
Epoch 1246/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7522 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0822 - val_sparse_categorical_accuracy: 0.5808
Epoch 1247/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7516 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0832 - val_sparse_categorical_accuracy: 0.5721
Epoch 1248/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7524 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0884 - val_sparse_categorical_accuracy: 0.5677
Epoch 1249/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7517 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0931 - val_sparse_categorical_accuracy: 0.5677
Epoch 1250/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7541 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0912 - val_sparse_categorical_accuracy: 0.5677
Epoch 1251/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7519 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0860 - val_sparse_categorical_accuracy: 0.5590
Epoch 1252/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7531 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.1010 - val_sparse_categorical_accuracy: 0.5633
Epoch 1253/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7502 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1053 - val_sparse_categorical_accuracy: 0.5677
Epoch 1254/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7499 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0974 - val_sparse_categorical_accuracy: 0.5808
Epoch 1255/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7503 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0916 - val_sparse_categorical_accuracy: 0.5721
Epoch 1256/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7519 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0968 - val_sparse_categorical_accuracy: 0.5808
Epoch 1257/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7520 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0911 - val_sparse_categorical_accuracy: 0.5764
Epoch 1258/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7513 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0842 - val_sparse_categorical_accuracy: 0.5677
Epoch 1259/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7506 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0932 - val_sparse_categorical_accuracy: 0.5677
Epoch 1260/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7513 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1028 - val_sparse_categorical_accuracy: 0.5677
Epoch 1261/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7505 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1083 - val_sparse_categorical_accuracy: 0.5415
Epoch 1262/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7516 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0880 - val_sparse_categorical_accuracy: 0.5633
Epoch 1263/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7525 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0832 - val_sparse_categorical_accuracy: 0.5459
Epoch 1264/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7576 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0941 - val_sparse_categorical_accuracy: 0.5808
Epoch 1265/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7524 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1180 - val_sparse_categorical_accuracy: 0.5328
Epoch 1266/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7592 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0843 - val_sparse_categorical_accuracy: 0.5721
Epoch 1267/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7500 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0778 - val_sparse_categorical_accuracy: 0.5459
Epoch 1268/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7548 - sparse_categorical_accuracy: 0.6701 - val_loss: 1.0970 - val_sparse_categorical_accuracy: 0.5677
Epoch 1269/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7537 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1192 - val_sparse_categorical_accuracy: 0.5240
Epoch 1270/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7553 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0871 - val_sparse_categorical_accuracy: 0.5764
Epoch 1271/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7547 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0825 - val_sparse_categorical_accuracy: 0.5764
Epoch 1272/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7541 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.1079 - val_sparse_categorical_accuracy: 0.5459
Epoch 1273/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7607 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1300 - val_sparse_categorical_accuracy: 0.5328
Epoch 1274/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7577 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0970 - val_sparse_categorical_accuracy: 0.5677
Epoch 1275/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7595 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0834 - val_sparse_categorical_accuracy: 0.5721
Epoch 1276/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7487 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1059 - val_sparse_categorical_accuracy: 0.5371
Epoch 1277/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7606 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0835 - val_sparse_categorical_accuracy: 0.5721
Epoch 1278/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7515 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0657 - val_sparse_categorical_accuracy: 0.5721
Epoch 1279/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7541 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0840 - val_sparse_categorical_accuracy: 0.5808
Epoch 1280/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7511 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0998 - val_sparse_categorical_accuracy: 0.5808
Epoch 1281/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7497 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.0986 - val_sparse_categorical_accuracy: 0.5808
Epoch 1282/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7535 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.0871 - val_sparse_categorical_accuracy: 0.5764
Epoch 1283/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7514 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0796 - val_sparse_categorical_accuracy: 0.5721
Epoch 1284/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7485 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0848 - val_sparse_categorical_accuracy: 0.5983
Epoch 1285/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7532 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0819 - val_sparse_categorical_accuracy: 0.5808
Epoch 1286/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7485 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0708 - val_sparse_categorical_accuracy: 0.5721
Epoch 1287/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7479 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0781 - val_sparse_categorical_accuracy: 0.5721
Epoch 1288/2000
2/2 [==============================] - 0s 21ms/step - loss: 0.7473 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0950 - val_sparse_categorical_accuracy: 0.5852
Epoch 1289/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7480 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1010 - val_sparse_categorical_accuracy: 0.5808
Epoch 1290/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7479 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0859 - val_sparse_categorical_accuracy: 0.5721
Epoch 1291/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7466 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0851 - val_sparse_categorical_accuracy: 0.5939
Epoch 1292/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7481 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0930 - val_sparse_categorical_accuracy: 0.5852
Epoch 1293/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7477 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0964 - val_sparse_categorical_accuracy: 0.5939
Epoch 1294/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7458 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0973 - val_sparse_categorical_accuracy: 0.5852
Epoch 1295/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7490 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0858 - val_sparse_categorical_accuracy: 0.5808
Epoch 1296/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7465 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0828 - val_sparse_categorical_accuracy: 0.5677
Epoch 1297/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7487 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0892 - val_sparse_categorical_accuracy: 0.5721
Epoch 1298/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7463 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0897 - val_sparse_categorical_accuracy: 0.5677
Epoch 1299/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7460 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0888 - val_sparse_categorical_accuracy: 0.5808
Epoch 1300/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7462 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0906 - val_sparse_categorical_accuracy: 0.5721
Epoch 1301/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7455 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0896 - val_sparse_categorical_accuracy: 0.5677
Epoch 1302/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7448 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0919 - val_sparse_categorical_accuracy: 0.5677
Epoch 1303/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7457 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0852 - val_sparse_categorical_accuracy: 0.5677
Epoch 1304/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7452 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0869 - val_sparse_categorical_accuracy: 0.5677
Epoch 1305/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7448 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1016 - val_sparse_categorical_accuracy: 0.5808
Epoch 1306/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.7445 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1053 - val_sparse_categorical_accuracy: 0.5764
Epoch 1307/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7481 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0933 - val_sparse_categorical_accuracy: 0.5895
Epoch 1308/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7480 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0877 - val_sparse_categorical_accuracy: 0.5633
Epoch 1309/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7503 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0912 - val_sparse_categorical_accuracy: 0.5721
Epoch 1310/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7486 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.0934 - val_sparse_categorical_accuracy: 0.5677
Epoch 1311/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7494 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0877 - val_sparse_categorical_accuracy: 0.5677
Epoch 1312/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7462 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1002 - val_sparse_categorical_accuracy: 0.5721
Epoch 1313/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7492 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1035 - val_sparse_categorical_accuracy: 0.5808
Epoch 1314/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7442 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1027 - val_sparse_categorical_accuracy: 0.5808
Epoch 1315/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7465 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0900 - val_sparse_categorical_accuracy: 0.5677
Epoch 1316/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7452 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0819 - val_sparse_categorical_accuracy: 0.5852
Epoch 1317/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7477 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0798 - val_sparse_categorical_accuracy: 0.5808
Epoch 1318/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7461 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1013 - val_sparse_categorical_accuracy: 0.5590
Epoch 1319/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7525 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1027 - val_sparse_categorical_accuracy: 0.5677
Epoch 1320/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7460 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0850 - val_sparse_categorical_accuracy: 0.5633
Epoch 1321/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7480 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0794 - val_sparse_categorical_accuracy: 0.5721
Epoch 1322/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.7516 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0767 - val_sparse_categorical_accuracy: 0.5721
Epoch 1323/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.7448 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0960 - val_sparse_categorical_accuracy: 0.5939
Epoch 1324/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.7504 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1053 - val_sparse_categorical_accuracy: 0.5852
Epoch 1325/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7459 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0893 - val_sparse_categorical_accuracy: 0.5721
Epoch 1326/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7495 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0778 - val_sparse_categorical_accuracy: 0.5852
Epoch 1327/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7454 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0983 - val_sparse_categorical_accuracy: 0.5240
Epoch 1328/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7531 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1003 - val_sparse_categorical_accuracy: 0.5240
Epoch 1329/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7463 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0831 - val_sparse_categorical_accuracy: 0.5721
Epoch 1330/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7532 - sparse_categorical_accuracy: 0.6774 - val_loss: 1.0851 - val_sparse_categorical_accuracy: 0.5721
Epoch 1331/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7467 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1161 - val_sparse_categorical_accuracy: 0.5415
Epoch 1332/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7538 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1184 - val_sparse_categorical_accuracy: 0.5502
Epoch 1333/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7504 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0964 - val_sparse_categorical_accuracy: 0.5590
Epoch 1334/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7504 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0990 - val_sparse_categorical_accuracy: 0.5764
Epoch 1335/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7448 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1091 - val_sparse_categorical_accuracy: 0.5633
Epoch 1336/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7483 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1123 - val_sparse_categorical_accuracy: 0.5590
Epoch 1337/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7424 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1030 - val_sparse_categorical_accuracy: 0.5590
Epoch 1338/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7555 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0953 - val_sparse_categorical_accuracy: 0.5677
Epoch 1339/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7461 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1198 - val_sparse_categorical_accuracy: 0.5502
Epoch 1340/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7503 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1210 - val_sparse_categorical_accuracy: 0.5459
Epoch 1341/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7496 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0944 - val_sparse_categorical_accuracy: 0.5677
Epoch 1342/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7440 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0895 - val_sparse_categorical_accuracy: 0.5677
Epoch 1343/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7445 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0991 - val_sparse_categorical_accuracy: 0.5808
Epoch 1344/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7425 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1048 - val_sparse_categorical_accuracy: 0.5808
Epoch 1345/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7444 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1022 - val_sparse_categorical_accuracy: 0.5895
Epoch 1346/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7423 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0966 - val_sparse_categorical_accuracy: 0.5808
Epoch 1347/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7417 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0970 - val_sparse_categorical_accuracy: 0.5895
Epoch 1348/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7442 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1054 - val_sparse_categorical_accuracy: 0.5852
Epoch 1349/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7455 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0970 - val_sparse_categorical_accuracy: 0.5852
Epoch 1350/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7431 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0767 - val_sparse_categorical_accuracy: 0.5808
Epoch 1351/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.7432 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0843 - val_sparse_categorical_accuracy: 0.5677
Epoch 1352/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7432 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1015 - val_sparse_categorical_accuracy: 0.5721
Epoch 1353/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7431 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1021 - val_sparse_categorical_accuracy: 0.5764
Epoch 1354/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7425 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1039 - val_sparse_categorical_accuracy: 0.5677
Epoch 1355/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7400 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1093 - val_sparse_categorical_accuracy: 0.5546
Epoch 1356/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7446 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1004 - val_sparse_categorical_accuracy: 0.5677
Epoch 1357/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7447 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0956 - val_sparse_categorical_accuracy: 0.5852
Epoch 1358/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7412 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1023 - val_sparse_categorical_accuracy: 0.5764
Epoch 1359/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7406 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1007 - val_sparse_categorical_accuracy: 0.5808
Epoch 1360/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7408 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0967 - val_sparse_categorical_accuracy: 0.5852
Epoch 1361/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7391 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1114 - val_sparse_categorical_accuracy: 0.5764
Epoch 1362/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7430 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1216 - val_sparse_categorical_accuracy: 0.5721
Epoch 1363/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7446 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1025 - val_sparse_categorical_accuracy: 0.5764
Epoch 1364/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7399 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0847 - val_sparse_categorical_accuracy: 0.5808
Epoch 1365/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7429 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0876 - val_sparse_categorical_accuracy: 0.5677
Epoch 1366/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7415 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1065 - val_sparse_categorical_accuracy: 0.5459
Epoch 1367/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7443 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1042 - val_sparse_categorical_accuracy: 0.5721
Epoch 1368/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7405 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1071 - val_sparse_categorical_accuracy: 0.5852
Epoch 1369/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7435 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1004 - val_sparse_categorical_accuracy: 0.5677
Epoch 1370/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7427 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1085 - val_sparse_categorical_accuracy: 0.5415
Epoch 1371/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7490 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0835 - val_sparse_categorical_accuracy: 0.5808
Epoch 1372/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7415 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0931 - val_sparse_categorical_accuracy: 0.5808
Epoch 1373/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7418 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1132 - val_sparse_categorical_accuracy: 0.5852
Epoch 1374/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7413 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1111 - val_sparse_categorical_accuracy: 0.5721
Epoch 1375/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.7425 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.0941 - val_sparse_categorical_accuracy: 0.5852
Epoch 1376/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7398 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1004 - val_sparse_categorical_accuracy: 0.5808
Epoch 1377/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7395 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1051 - val_sparse_categorical_accuracy: 0.5852
Epoch 1378/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7411 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1016 - val_sparse_categorical_accuracy: 0.6026
Epoch 1379/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7401 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0925 - val_sparse_categorical_accuracy: 0.5852
Epoch 1380/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7396 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.0955 - val_sparse_categorical_accuracy: 0.5764
Epoch 1381/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7372 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1135 - val_sparse_categorical_accuracy: 0.5895
Epoch 1382/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7398 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1100 - val_sparse_categorical_accuracy: 0.5721
Epoch 1383/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7402 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0923 - val_sparse_categorical_accuracy: 0.5677
Epoch 1384/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7411 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.0860 - val_sparse_categorical_accuracy: 0.5852
Epoch 1385/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7399 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1052 - val_sparse_categorical_accuracy: 0.5808
Epoch 1386/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7423 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1013 - val_sparse_categorical_accuracy: 0.5677
Epoch 1387/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7380 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.0856 - val_sparse_categorical_accuracy: 0.5677
Epoch 1388/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7450 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.0960 - val_sparse_categorical_accuracy: 0.5764
Epoch 1389/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7410 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1186 - val_sparse_categorical_accuracy: 0.5502
Epoch 1390/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7424 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0989 - val_sparse_categorical_accuracy: 0.5808
Epoch 1391/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7374 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0882 - val_sparse_categorical_accuracy: 0.5721
Epoch 1392/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7372 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0906 - val_sparse_categorical_accuracy: 0.5590
Epoch 1393/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7389 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0980 - val_sparse_categorical_accuracy: 0.5590
Epoch 1394/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7367 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1058 - val_sparse_categorical_accuracy: 0.5590
Epoch 1395/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7387 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1085 - val_sparse_categorical_accuracy: 0.5633
Epoch 1396/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7379 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1139 - val_sparse_categorical_accuracy: 0.5502
Epoch 1397/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7378 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1067 - val_sparse_categorical_accuracy: 0.5459
Epoch 1398/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7372 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1070 - val_sparse_categorical_accuracy: 0.5590
Epoch 1399/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7357 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1002 - val_sparse_categorical_accuracy: 0.5590
Epoch 1400/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7379 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.0876 - val_sparse_categorical_accuracy: 0.5852
Epoch 1401/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7370 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.0871 - val_sparse_categorical_accuracy: 0.5764
Epoch 1402/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7362 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0972 - val_sparse_categorical_accuracy: 0.5677
Epoch 1403/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7380 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1067 - val_sparse_categorical_accuracy: 0.5590
Epoch 1404/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7345 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1021 - val_sparse_categorical_accuracy: 0.5852
Epoch 1405/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7371 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.0973 - val_sparse_categorical_accuracy: 0.5764
Epoch 1406/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7352 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1025 - val_sparse_categorical_accuracy: 0.5546
Epoch 1407/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7393 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0992 - val_sparse_categorical_accuracy: 0.5677
Epoch 1408/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7383 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1025 - val_sparse_categorical_accuracy: 0.5764
Epoch 1409/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7352 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1142 - val_sparse_categorical_accuracy: 0.5721
Epoch 1410/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7381 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1123 - val_sparse_categorical_accuracy: 0.5677
Epoch 1411/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7378 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1087 - val_sparse_categorical_accuracy: 0.5677
Epoch 1412/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7359 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1009 - val_sparse_categorical_accuracy: 0.5764
Epoch 1413/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7390 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1065 - val_sparse_categorical_accuracy: 0.5677
Epoch 1414/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7347 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1150 - val_sparse_categorical_accuracy: 0.5633
Epoch 1415/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7345 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1223 - val_sparse_categorical_accuracy: 0.5633
Epoch 1416/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7340 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1093 - val_sparse_categorical_accuracy: 0.5590
Epoch 1417/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7333 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0971 - val_sparse_categorical_accuracy: 0.5677
Epoch 1418/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7322 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1017 - val_sparse_categorical_accuracy: 0.5808
Epoch 1419/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7346 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0991 - val_sparse_categorical_accuracy: 0.5721
Epoch 1420/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7335 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0883 - val_sparse_categorical_accuracy: 0.5852
Epoch 1421/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7356 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0859 - val_sparse_categorical_accuracy: 0.5721
Epoch 1422/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7350 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1037 - val_sparse_categorical_accuracy: 0.5764
Epoch 1423/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7350 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1174 - val_sparse_categorical_accuracy: 0.5677
Epoch 1424/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7336 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0936 - val_sparse_categorical_accuracy: 0.5808
Epoch 1425/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7387 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0905 - val_sparse_categorical_accuracy: 0.5764
Epoch 1426/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7377 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1252 - val_sparse_categorical_accuracy: 0.5371
Epoch 1427/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7356 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1244 - val_sparse_categorical_accuracy: 0.5590
Epoch 1428/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7326 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1079 - val_sparse_categorical_accuracy: 0.5677
Epoch 1429/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7369 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1049 - val_sparse_categorical_accuracy: 0.5808
Epoch 1430/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7343 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1227 - val_sparse_categorical_accuracy: 0.5415
Epoch 1431/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7383 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1072 - val_sparse_categorical_accuracy: 0.5764
Epoch 1432/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7370 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0959 - val_sparse_categorical_accuracy: 0.5721
Epoch 1433/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7354 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1104 - val_sparse_categorical_accuracy: 0.5721
Epoch 1434/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7412 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1164 - val_sparse_categorical_accuracy: 0.5633
Epoch 1435/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7392 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.0999 - val_sparse_categorical_accuracy: 0.5852
Epoch 1436/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7338 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1146 - val_sparse_categorical_accuracy: 0.5852
Epoch 1437/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7340 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1198 - val_sparse_categorical_accuracy: 0.5808
Epoch 1438/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7357 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1018 - val_sparse_categorical_accuracy: 0.5895
Epoch 1439/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7333 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0880 - val_sparse_categorical_accuracy: 0.5721
Epoch 1440/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7351 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0986 - val_sparse_categorical_accuracy: 0.5764
Epoch 1441/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7339 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1262 - val_sparse_categorical_accuracy: 0.5590
Epoch 1442/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7353 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1256 - val_sparse_categorical_accuracy: 0.5677
Epoch 1443/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7320 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1063 - val_sparse_categorical_accuracy: 0.5808
Epoch 1444/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7318 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.0879 - val_sparse_categorical_accuracy: 0.5852
Epoch 1445/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7335 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.0842 - val_sparse_categorical_accuracy: 0.5764
Epoch 1446/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7360 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1023 - val_sparse_categorical_accuracy: 0.5895
Epoch 1447/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7290 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1363 - val_sparse_categorical_accuracy: 0.5415
Epoch 1448/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7389 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1145 - val_sparse_categorical_accuracy: 0.5764
Epoch 1449/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7325 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1048 - val_sparse_categorical_accuracy: 0.5808
Epoch 1450/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7407 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1119 - val_sparse_categorical_accuracy: 0.5852
Epoch 1451/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7318 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1274 - val_sparse_categorical_accuracy: 0.5502
Epoch 1452/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7331 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1186 - val_sparse_categorical_accuracy: 0.5721
Epoch 1453/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7328 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0995 - val_sparse_categorical_accuracy: 0.5633
Epoch 1454/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7340 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.0876 - val_sparse_categorical_accuracy: 0.5808
Epoch 1455/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7333 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.0936 - val_sparse_categorical_accuracy: 0.5764
Epoch 1456/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7300 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1216 - val_sparse_categorical_accuracy: 0.5808
Epoch 1457/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7302 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1187 - val_sparse_categorical_accuracy: 0.5721
Epoch 1458/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7303 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1052 - val_sparse_categorical_accuracy: 0.5808
Epoch 1459/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7322 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1062 - val_sparse_categorical_accuracy: 0.5808
Epoch 1460/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7301 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1158 - val_sparse_categorical_accuracy: 0.5764
Epoch 1461/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7294 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1175 - val_sparse_categorical_accuracy: 0.5895
Epoch 1462/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7289 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1259 - val_sparse_categorical_accuracy: 0.5895
Epoch 1463/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7338 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1213 - val_sparse_categorical_accuracy: 0.5764
Epoch 1464/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7290 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1101 - val_sparse_categorical_accuracy: 0.5721
Epoch 1465/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7327 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1184 - val_sparse_categorical_accuracy: 0.5764
Epoch 1466/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7292 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1515 - val_sparse_categorical_accuracy: 0.5197
Epoch 1467/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7403 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1293 - val_sparse_categorical_accuracy: 0.5677
Epoch 1468/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.7362 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1133 - val_sparse_categorical_accuracy: 0.5633
Epoch 1469/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7338 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1174 - val_sparse_categorical_accuracy: 0.5677
Epoch 1470/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7367 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1150 - val_sparse_categorical_accuracy: 0.5808
Epoch 1471/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7343 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.0916 - val_sparse_categorical_accuracy: 0.5852
Epoch 1472/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7343 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.0989 - val_sparse_categorical_accuracy: 0.5808
Epoch 1473/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7304 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1320 - val_sparse_categorical_accuracy: 0.5721
Epoch 1474/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7359 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1236 - val_sparse_categorical_accuracy: 0.5764
Epoch 1475/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7311 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1054 - val_sparse_categorical_accuracy: 0.5546
Epoch 1476/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7341 - sparse_categorical_accuracy: 0.6788 - val_loss: 1.0968 - val_sparse_categorical_accuracy: 0.5852
Epoch 1477/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7300 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1243 - val_sparse_categorical_accuracy: 0.5459
Epoch 1478/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7318 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1489 - val_sparse_categorical_accuracy: 0.5502
Epoch 1479/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7339 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1393 - val_sparse_categorical_accuracy: 0.5721
Epoch 1480/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7310 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1217 - val_sparse_categorical_accuracy: 0.5677
Epoch 1481/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7285 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1091 - val_sparse_categorical_accuracy: 0.5721
Epoch 1482/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7269 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1130 - val_sparse_categorical_accuracy: 0.5852
Epoch 1483/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7297 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1110 - val_sparse_categorical_accuracy: 0.5939
Epoch 1484/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7289 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1072 - val_sparse_categorical_accuracy: 0.5852
Epoch 1485/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7281 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1262 - val_sparse_categorical_accuracy: 0.5633
Epoch 1486/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7308 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1373 - val_sparse_categorical_accuracy: 0.5415
Epoch 1487/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7305 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1219 - val_sparse_categorical_accuracy: 0.5677
Epoch 1488/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7279 - sparse_categorical_accuracy: 0.6803 - val_loss: 1.1105 - val_sparse_categorical_accuracy: 0.5677
Epoch 1489/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7277 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1156 - val_sparse_categorical_accuracy: 0.5633
Epoch 1490/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7300 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1260 - val_sparse_categorical_accuracy: 0.5633
Epoch 1491/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7285 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1294 - val_sparse_categorical_accuracy: 0.5939
Epoch 1492/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7309 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1222 - val_sparse_categorical_accuracy: 0.5983
Epoch 1493/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7308 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1105 - val_sparse_categorical_accuracy: 0.5764
Epoch 1494/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7265 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1079 - val_sparse_categorical_accuracy: 0.5677
Epoch 1495/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7321 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1148 - val_sparse_categorical_accuracy: 0.5808
Epoch 1496/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7283 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1317 - val_sparse_categorical_accuracy: 0.5764
Epoch 1497/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7261 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1348 - val_sparse_categorical_accuracy: 0.5721
Epoch 1498/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7277 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1115 - val_sparse_categorical_accuracy: 0.5721
Epoch 1499/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7271 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1006 - val_sparse_categorical_accuracy: 0.5764
Epoch 1500/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7314 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1013 - val_sparse_categorical_accuracy: 0.6070
Epoch 1501/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7272 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1169 - val_sparse_categorical_accuracy: 0.5852
Epoch 1502/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7300 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1250 - val_sparse_categorical_accuracy: 0.5808
Epoch 1503/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7266 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1140 - val_sparse_categorical_accuracy: 0.5808
Epoch 1504/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7298 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.0975 - val_sparse_categorical_accuracy: 0.5633
Epoch 1505/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7288 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1066 - val_sparse_categorical_accuracy: 0.5808
Epoch 1506/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7327 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1146 - val_sparse_categorical_accuracy: 0.5852
Epoch 1507/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7281 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1228 - val_sparse_categorical_accuracy: 0.5764
Epoch 1508/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7261 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1225 - val_sparse_categorical_accuracy: 0.5764
Epoch 1509/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7259 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1159 - val_sparse_categorical_accuracy: 0.5852
Epoch 1510/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7250 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1177 - val_sparse_categorical_accuracy: 0.5895
Epoch 1511/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7266 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1157 - val_sparse_categorical_accuracy: 0.5808
Epoch 1512/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7229 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1323 - val_sparse_categorical_accuracy: 0.5808
Epoch 1513/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7246 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1362 - val_sparse_categorical_accuracy: 0.5721
Epoch 1514/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7244 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1166 - val_sparse_categorical_accuracy: 0.5852
Epoch 1515/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7263 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1111 - val_sparse_categorical_accuracy: 0.5939
Epoch 1516/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7280 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1210 - val_sparse_categorical_accuracy: 0.5677
Epoch 1517/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7241 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1279 - val_sparse_categorical_accuracy: 0.5677
Epoch 1518/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7235 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1280 - val_sparse_categorical_accuracy: 0.5677
Epoch 1519/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7273 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1251 - val_sparse_categorical_accuracy: 0.5939
Epoch 1520/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7245 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1318 - val_sparse_categorical_accuracy: 0.5633
Epoch 1521/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7252 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1274 - val_sparse_categorical_accuracy: 0.5808
Epoch 1522/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7267 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1133 - val_sparse_categorical_accuracy: 0.5852
Epoch 1523/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7253 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1128 - val_sparse_categorical_accuracy: 0.5764
Epoch 1524/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7307 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1126 - val_sparse_categorical_accuracy: 0.5677
Epoch 1525/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7279 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1189 - val_sparse_categorical_accuracy: 0.5895
Epoch 1526/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7304 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1215 - val_sparse_categorical_accuracy: 0.5808
Epoch 1527/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7245 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1201 - val_sparse_categorical_accuracy: 0.5721
Epoch 1528/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7289 - sparse_categorical_accuracy: 0.6759 - val_loss: 1.1206 - val_sparse_categorical_accuracy: 0.5633
Epoch 1529/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7266 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1284 - val_sparse_categorical_accuracy: 0.5677
Epoch 1530/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7267 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1343 - val_sparse_categorical_accuracy: 0.5677
Epoch 1531/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.7237 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1118 - val_sparse_categorical_accuracy: 0.5677
Epoch 1532/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7265 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1084 - val_sparse_categorical_accuracy: 0.5721
Epoch 1533/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7242 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1340 - val_sparse_categorical_accuracy: 0.5415
Epoch 1534/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7303 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1253 - val_sparse_categorical_accuracy: 0.5852
Epoch 1535/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7221 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1253 - val_sparse_categorical_accuracy: 0.5721
Epoch 1536/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7361 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1366 - val_sparse_categorical_accuracy: 0.5852
Epoch 1537/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7243 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1553 - val_sparse_categorical_accuracy: 0.5459
Epoch 1538/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7318 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1179 - val_sparse_categorical_accuracy: 0.5895
Epoch 1539/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7237 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1111 - val_sparse_categorical_accuracy: 0.5633
Epoch 1540/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7321 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1281 - val_sparse_categorical_accuracy: 0.5808
Epoch 1541/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7226 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1548 - val_sparse_categorical_accuracy: 0.5240
Epoch 1542/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7271 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1323 - val_sparse_categorical_accuracy: 0.5677
Epoch 1543/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7216 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1226 - val_sparse_categorical_accuracy: 0.5633
Epoch 1544/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7285 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1339 - val_sparse_categorical_accuracy: 0.5764
Epoch 1545/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7257 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1359 - val_sparse_categorical_accuracy: 0.5721
Epoch 1546/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7237 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1127 - val_sparse_categorical_accuracy: 0.5852
Epoch 1547/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7214 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1187 - val_sparse_categorical_accuracy: 0.5808
Epoch 1548/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.7221 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1364 - val_sparse_categorical_accuracy: 0.5764
Epoch 1549/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.7211 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1517 - val_sparse_categorical_accuracy: 0.5721
Epoch 1550/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.7239 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1348 - val_sparse_categorical_accuracy: 0.5721
Epoch 1551/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.7235 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1224 - val_sparse_categorical_accuracy: 0.5764
Epoch 1552/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7226 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1212 - val_sparse_categorical_accuracy: 0.5721
Epoch 1553/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7262 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1080 - val_sparse_categorical_accuracy: 0.5764
Epoch 1554/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7224 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1169 - val_sparse_categorical_accuracy: 0.5764
Epoch 1555/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7236 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1329 - val_sparse_categorical_accuracy: 0.5808
Epoch 1556/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7215 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1390 - val_sparse_categorical_accuracy: 0.5677
Epoch 1557/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7288 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1195 - val_sparse_categorical_accuracy: 0.5852
Epoch 1558/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7226 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1155 - val_sparse_categorical_accuracy: 0.5852
Epoch 1559/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7212 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1268 - val_sparse_categorical_accuracy: 0.5852
Epoch 1560/2000
2/2 [==============================] - 0s 23ms/step - loss: 0.7202 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1262 - val_sparse_categorical_accuracy: 0.5895
Epoch 1561/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7213 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1191 - val_sparse_categorical_accuracy: 0.5808
Epoch 1562/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7236 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1299 - val_sparse_categorical_accuracy: 0.5808
Epoch 1563/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7254 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1352 - val_sparse_categorical_accuracy: 0.5590
Epoch 1564/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7216 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1238 - val_sparse_categorical_accuracy: 0.5764
Epoch 1565/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7257 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1198 - val_sparse_categorical_accuracy: 0.5808
Epoch 1566/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7215 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1279 - val_sparse_categorical_accuracy: 0.5808
Epoch 1567/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7228 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1349 - val_sparse_categorical_accuracy: 0.5939
Epoch 1568/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7226 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1366 - val_sparse_categorical_accuracy: 0.5852
Epoch 1569/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7224 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1197 - val_sparse_categorical_accuracy: 0.5983
Epoch 1570/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7219 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1168 - val_sparse_categorical_accuracy: 0.5808
Epoch 1571/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7219 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1201 - val_sparse_categorical_accuracy: 0.5764
Epoch 1572/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7238 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1270 - val_sparse_categorical_accuracy: 0.5852
Epoch 1573/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7214 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1336 - val_sparse_categorical_accuracy: 0.5808
Epoch 1574/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7216 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1247 - val_sparse_categorical_accuracy: 0.5852
Epoch 1575/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7202 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1255 - val_sparse_categorical_accuracy: 0.5939
Epoch 1576/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.7189 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1246 - val_sparse_categorical_accuracy: 0.5852
Epoch 1577/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7215 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1193 - val_sparse_categorical_accuracy: 0.5895
Epoch 1578/2000
2/2 [==============================] - 0s 23ms/step - loss: 0.7193 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1233 - val_sparse_categorical_accuracy: 0.5895
Epoch 1579/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7193 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1339 - val_sparse_categorical_accuracy: 0.5677
Epoch 1580/2000
2/2 [==============================] - 0s 27ms/step - loss: 0.7195 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1424 - val_sparse_categorical_accuracy: 0.5677
Epoch 1581/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7210 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1343 - val_sparse_categorical_accuracy: 0.5633
Epoch 1582/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.7207 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1136 - val_sparse_categorical_accuracy: 0.5721
Epoch 1583/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7200 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1122 - val_sparse_categorical_accuracy: 0.5852
Epoch 1584/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7192 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1312 - val_sparse_categorical_accuracy: 0.5852
Epoch 1585/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7192 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1353 - val_sparse_categorical_accuracy: 0.5895
Epoch 1586/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7196 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1291 - val_sparse_categorical_accuracy: 0.5852
Epoch 1587/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7207 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1264 - val_sparse_categorical_accuracy: 0.5764
Epoch 1588/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7191 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1294 - val_sparse_categorical_accuracy: 0.5764
Epoch 1589/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7192 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1298 - val_sparse_categorical_accuracy: 0.5677
Epoch 1590/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7189 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1215 - val_sparse_categorical_accuracy: 0.5808
Epoch 1591/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7176 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1375 - val_sparse_categorical_accuracy: 0.5764
Epoch 1592/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7200 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1458 - val_sparse_categorical_accuracy: 0.5546
Epoch 1593/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7194 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1194 - val_sparse_categorical_accuracy: 0.5677
Epoch 1594/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7196 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1067 - val_sparse_categorical_accuracy: 0.5677
Epoch 1595/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7233 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1197 - val_sparse_categorical_accuracy: 0.5721
Epoch 1596/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7184 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1428 - val_sparse_categorical_accuracy: 0.5895
Epoch 1597/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7190 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1429 - val_sparse_categorical_accuracy: 0.5939
Epoch 1598/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7177 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1242 - val_sparse_categorical_accuracy: 0.5852
Epoch 1599/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7185 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1251 - val_sparse_categorical_accuracy: 0.5721
Epoch 1600/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7197 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1395 - val_sparse_categorical_accuracy: 0.5764
Epoch 1601/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7173 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1462 - val_sparse_categorical_accuracy: 0.5808
Epoch 1602/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7200 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1350 - val_sparse_categorical_accuracy: 0.5895
Epoch 1603/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7189 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1329 - val_sparse_categorical_accuracy: 0.5852
Epoch 1604/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7183 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1495 - val_sparse_categorical_accuracy: 0.5677
Epoch 1605/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7203 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1435 - val_sparse_categorical_accuracy: 0.5721
Epoch 1606/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7170 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1211 - val_sparse_categorical_accuracy: 0.5721
Epoch 1607/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7184 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1268 - val_sparse_categorical_accuracy: 0.5677
Epoch 1608/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7181 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1575 - val_sparse_categorical_accuracy: 0.5721
Epoch 1609/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7187 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1611 - val_sparse_categorical_accuracy: 0.5764
Epoch 1610/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7188 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1319 - val_sparse_categorical_accuracy: 0.5895
Epoch 1611/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7207 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1241 - val_sparse_categorical_accuracy: 0.5721
Epoch 1612/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7225 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1351 - val_sparse_categorical_accuracy: 0.5677
Epoch 1613/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7160 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1450 - val_sparse_categorical_accuracy: 0.5721
Epoch 1614/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7226 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1589 - val_sparse_categorical_accuracy: 0.5677
Epoch 1615/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7200 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1448 - val_sparse_categorical_accuracy: 0.5546
Epoch 1616/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7208 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1204 - val_sparse_categorical_accuracy: 0.5764
Epoch 1617/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7173 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1244 - val_sparse_categorical_accuracy: 0.5764
Epoch 1618/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7152 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1365 - val_sparse_categorical_accuracy: 0.5939
Epoch 1619/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7158 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1334 - val_sparse_categorical_accuracy: 0.5983
Epoch 1620/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7169 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1282 - val_sparse_categorical_accuracy: 0.5808
Epoch 1621/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7166 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1486 - val_sparse_categorical_accuracy: 0.5808
Epoch 1622/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7197 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1462 - val_sparse_categorical_accuracy: 0.5764
Epoch 1623/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7157 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1245 - val_sparse_categorical_accuracy: 0.5808
Epoch 1624/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7223 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1260 - val_sparse_categorical_accuracy: 0.5721
Epoch 1625/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7158 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1569 - val_sparse_categorical_accuracy: 0.5371
Epoch 1626/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7265 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1566 - val_sparse_categorical_accuracy: 0.5459
Epoch 1627/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7200 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1251 - val_sparse_categorical_accuracy: 0.5852
Epoch 1628/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7235 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1156 - val_sparse_categorical_accuracy: 0.5852
Epoch 1629/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7179 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1493 - val_sparse_categorical_accuracy: 0.5459
Epoch 1630/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7222 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1429 - val_sparse_categorical_accuracy: 0.5721
Epoch 1631/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7159 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1197 - val_sparse_categorical_accuracy: 0.5502
Epoch 1632/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7208 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1253 - val_sparse_categorical_accuracy: 0.5721
Epoch 1633/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7151 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1527 - val_sparse_categorical_accuracy: 0.5459
Epoch 1634/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7195 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1388 - val_sparse_categorical_accuracy: 0.5852
Epoch 1635/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7151 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1296 - val_sparse_categorical_accuracy: 0.5677
Epoch 1636/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7188 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1413 - val_sparse_categorical_accuracy: 0.5808
Epoch 1637/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7173 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1579 - val_sparse_categorical_accuracy: 0.5633
Epoch 1638/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7184 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1418 - val_sparse_categorical_accuracy: 0.5808
Epoch 1639/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7164 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1410 - val_sparse_categorical_accuracy: 0.5895
Epoch 1640/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7144 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1475 - val_sparse_categorical_accuracy: 0.5852
Epoch 1641/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7152 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1324 - val_sparse_categorical_accuracy: 0.5808
Epoch 1642/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7135 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1244 - val_sparse_categorical_accuracy: 0.5721
Epoch 1643/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7154 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1361 - val_sparse_categorical_accuracy: 0.5721
Epoch 1644/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7128 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1486 - val_sparse_categorical_accuracy: 0.5808
Epoch 1645/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7144 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1388 - val_sparse_categorical_accuracy: 0.5808
Epoch 1646/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7160 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1323 - val_sparse_categorical_accuracy: 0.5852
Epoch 1647/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7152 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1407 - val_sparse_categorical_accuracy: 0.5808
Epoch 1648/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7149 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1440 - val_sparse_categorical_accuracy: 0.5721
Epoch 1649/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7177 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1246 - val_sparse_categorical_accuracy: 0.5721
Epoch 1650/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.7138 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1351 - val_sparse_categorical_accuracy: 0.5721
Epoch 1651/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7156 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1538 - val_sparse_categorical_accuracy: 0.5633
Epoch 1652/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7129 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1482 - val_sparse_categorical_accuracy: 0.5590
Epoch 1653/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7121 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1403 - val_sparse_categorical_accuracy: 0.5590
Epoch 1654/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7131 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1363 - val_sparse_categorical_accuracy: 0.5546
Epoch 1655/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7135 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1401 - val_sparse_categorical_accuracy: 0.5852
Epoch 1656/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7116 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1564 - val_sparse_categorical_accuracy: 0.5764
Epoch 1657/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7137 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1396 - val_sparse_categorical_accuracy: 0.5852
Epoch 1658/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7140 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1182 - val_sparse_categorical_accuracy: 0.6026
Epoch 1659/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7148 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1242 - val_sparse_categorical_accuracy: 0.5983
Epoch 1660/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7152 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1389 - val_sparse_categorical_accuracy: 0.5808
Epoch 1661/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7134 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1460 - val_sparse_categorical_accuracy: 0.5808
Epoch 1662/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7143 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1475 - val_sparse_categorical_accuracy: 0.5721
Epoch 1663/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7128 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1512 - val_sparse_categorical_accuracy: 0.5633
Epoch 1664/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7143 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1427 - val_sparse_categorical_accuracy: 0.5677
Epoch 1665/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7140 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1272 - val_sparse_categorical_accuracy: 0.5721
Epoch 1666/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7159 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1490 - val_sparse_categorical_accuracy: 0.5459
Epoch 1667/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7168 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1555 - val_sparse_categorical_accuracy: 0.5808
Epoch 1668/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7148 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1330 - val_sparse_categorical_accuracy: 0.5721
Epoch 1669/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7126 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1228 - val_sparse_categorical_accuracy: 0.5808
Epoch 1670/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7153 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1370 - val_sparse_categorical_accuracy: 0.5764
Epoch 1671/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7127 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1442 - val_sparse_categorical_accuracy: 0.5677
Epoch 1672/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7118 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1341 - val_sparse_categorical_accuracy: 0.5764
Epoch 1673/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7102 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1340 - val_sparse_categorical_accuracy: 0.5590
Epoch 1674/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7162 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1430 - val_sparse_categorical_accuracy: 0.5764
Epoch 1675/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7156 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1498 - val_sparse_categorical_accuracy: 0.5677
Epoch 1676/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7127 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1431 - val_sparse_categorical_accuracy: 0.5633
Epoch 1677/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7167 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1458 - val_sparse_categorical_accuracy: 0.5633
Epoch 1678/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7176 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1603 - val_sparse_categorical_accuracy: 0.5502
Epoch 1679/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7152 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1384 - val_sparse_categorical_accuracy: 0.5721
Epoch 1680/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7132 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1079 - val_sparse_categorical_accuracy: 0.5721
Epoch 1681/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7243 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1070 - val_sparse_categorical_accuracy: 0.5677
Epoch 1682/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7205 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1458 - val_sparse_categorical_accuracy: 0.5808
Epoch 1683/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7181 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1545 - val_sparse_categorical_accuracy: 0.5677
Epoch 1684/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7135 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1353 - val_sparse_categorical_accuracy: 0.5721
Epoch 1685/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7164 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1534 - val_sparse_categorical_accuracy: 0.5633
Epoch 1686/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7153 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1738 - val_sparse_categorical_accuracy: 0.5415
Epoch 1687/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7171 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1626 - val_sparse_categorical_accuracy: 0.5677
Epoch 1688/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7104 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1498 - val_sparse_categorical_accuracy: 0.5546
Epoch 1689/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7142 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1483 - val_sparse_categorical_accuracy: 0.5895
Epoch 1690/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7125 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1531 - val_sparse_categorical_accuracy: 0.5721
Epoch 1691/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7134 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1433 - val_sparse_categorical_accuracy: 0.5808
Epoch 1692/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7131 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1368 - val_sparse_categorical_accuracy: 0.5721
Epoch 1693/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7101 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1531 - val_sparse_categorical_accuracy: 0.5590
Epoch 1694/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7081 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1681 - val_sparse_categorical_accuracy: 0.5633
Epoch 1695/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7096 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1719 - val_sparse_categorical_accuracy: 0.5721
Epoch 1696/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7107 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1646 - val_sparse_categorical_accuracy: 0.5721
Epoch 1697/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7104 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1514 - val_sparse_categorical_accuracy: 0.5677
Epoch 1698/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7092 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1517 - val_sparse_categorical_accuracy: 0.5764
Epoch 1699/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7087 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1528 - val_sparse_categorical_accuracy: 0.5808
Epoch 1700/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7098 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1550 - val_sparse_categorical_accuracy: 0.5852
Epoch 1701/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7099 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1506 - val_sparse_categorical_accuracy: 0.5764
Epoch 1702/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7110 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1449 - val_sparse_categorical_accuracy: 0.5677
Epoch 1703/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7088 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1570 - val_sparse_categorical_accuracy: 0.5677
Epoch 1704/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7098 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1433 - val_sparse_categorical_accuracy: 0.5808
Epoch 1705/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7108 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1450 - val_sparse_categorical_accuracy: 0.5764
Epoch 1706/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7125 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1623 - val_sparse_categorical_accuracy: 0.5502
Epoch 1707/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7107 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1543 - val_sparse_categorical_accuracy: 0.5590
Epoch 1708/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7088 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1534 - val_sparse_categorical_accuracy: 0.5546
Epoch 1709/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7108 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1553 - val_sparse_categorical_accuracy: 0.5677
Epoch 1710/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7099 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1657 - val_sparse_categorical_accuracy: 0.5502
Epoch 1711/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7103 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1604 - val_sparse_categorical_accuracy: 0.5721
Epoch 1712/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7100 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1445 - val_sparse_categorical_accuracy: 0.5852
Epoch 1713/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7084 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1428 - val_sparse_categorical_accuracy: 0.5852
Epoch 1714/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7074 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1495 - val_sparse_categorical_accuracy: 0.5808
Epoch 1715/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7080 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1577 - val_sparse_categorical_accuracy: 0.5808
Epoch 1716/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7081 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1529 - val_sparse_categorical_accuracy: 0.5895
Epoch 1717/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7083 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1626 - val_sparse_categorical_accuracy: 0.5852
Epoch 1718/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7078 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1723 - val_sparse_categorical_accuracy: 0.5633
Epoch 1719/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7080 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1582 - val_sparse_categorical_accuracy: 0.5633
Epoch 1720/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7062 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1504 - val_sparse_categorical_accuracy: 0.5546
Epoch 1721/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7072 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1466 - val_sparse_categorical_accuracy: 0.5633
Epoch 1722/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7066 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1542 - val_sparse_categorical_accuracy: 0.5677
Epoch 1723/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7079 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1619 - val_sparse_categorical_accuracy: 0.5764
Epoch 1724/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7072 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1525 - val_sparse_categorical_accuracy: 0.5677
Epoch 1725/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7066 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1541 - val_sparse_categorical_accuracy: 0.5721
Epoch 1726/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7076 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1617 - val_sparse_categorical_accuracy: 0.5677
Epoch 1727/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7077 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1555 - val_sparse_categorical_accuracy: 0.5721
Epoch 1728/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7091 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1430 - val_sparse_categorical_accuracy: 0.5808
Epoch 1729/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7081 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1550 - val_sparse_categorical_accuracy: 0.5677
Epoch 1730/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7101 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1636 - val_sparse_categorical_accuracy: 0.5677
Epoch 1731/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7090 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1529 - val_sparse_categorical_accuracy: 0.5721
Epoch 1732/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7100 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1481 - val_sparse_categorical_accuracy: 0.5764
Epoch 1733/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7076 - sparse_categorical_accuracy: 0.6847 - val_loss: 1.1775 - val_sparse_categorical_accuracy: 0.5590
Epoch 1734/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7101 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1900 - val_sparse_categorical_accuracy: 0.5590
Epoch 1735/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7087 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1591 - val_sparse_categorical_accuracy: 0.5633
Epoch 1736/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7086 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1533 - val_sparse_categorical_accuracy: 0.5721
Epoch 1737/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7064 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1659 - val_sparse_categorical_accuracy: 0.5546
Epoch 1738/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7060 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1536 - val_sparse_categorical_accuracy: 0.5677
Epoch 1739/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7059 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1524 - val_sparse_categorical_accuracy: 0.5764
Epoch 1740/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7085 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1612 - val_sparse_categorical_accuracy: 0.5852
Epoch 1741/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7059 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1454 - val_sparse_categorical_accuracy: 0.5808
Epoch 1742/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7063 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1419 - val_sparse_categorical_accuracy: 0.5852
Epoch 1743/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7068 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1649 - val_sparse_categorical_accuracy: 0.5633
Epoch 1744/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7064 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1785 - val_sparse_categorical_accuracy: 0.5590
Epoch 1745/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.7066 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1526 - val_sparse_categorical_accuracy: 0.5764
Epoch 1746/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7066 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1526 - val_sparse_categorical_accuracy: 0.5808
Epoch 1747/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7077 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1713 - val_sparse_categorical_accuracy: 0.5633
Epoch 1748/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7063 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1657 - val_sparse_categorical_accuracy: 0.5764
Epoch 1749/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7041 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1633 - val_sparse_categorical_accuracy: 0.5546
Epoch 1750/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7080 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1580 - val_sparse_categorical_accuracy: 0.5721
Epoch 1751/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7047 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1624 - val_sparse_categorical_accuracy: 0.5590
Epoch 1752/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7082 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1500 - val_sparse_categorical_accuracy: 0.5677
Epoch 1753/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7046 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1458 - val_sparse_categorical_accuracy: 0.5764
Epoch 1754/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7066 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1414 - val_sparse_categorical_accuracy: 0.5808
Epoch 1755/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7051 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.1476 - val_sparse_categorical_accuracy: 0.5808
Epoch 1756/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.7071 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1633 - val_sparse_categorical_accuracy: 0.5721
Epoch 1757/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7058 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1711 - val_sparse_categorical_accuracy: 0.5371
Epoch 1758/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7079 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1624 - val_sparse_categorical_accuracy: 0.5590
Epoch 1759/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7082 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1550 - val_sparse_categorical_accuracy: 0.5721
Epoch 1760/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7058 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1540 - val_sparse_categorical_accuracy: 0.5590
Epoch 1761/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7066 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1573 - val_sparse_categorical_accuracy: 0.5633
Epoch 1762/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7060 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1626 - val_sparse_categorical_accuracy: 0.5590
Epoch 1763/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7040 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1619 - val_sparse_categorical_accuracy: 0.5764
Epoch 1764/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7065 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1498 - val_sparse_categorical_accuracy: 0.5895
Epoch 1765/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7071 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1537 - val_sparse_categorical_accuracy: 0.5721
Epoch 1766/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7055 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1711 - val_sparse_categorical_accuracy: 0.5721
Epoch 1767/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7047 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1732 - val_sparse_categorical_accuracy: 0.5590
Epoch 1768/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7098 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1585 - val_sparse_categorical_accuracy: 0.5808
Epoch 1769/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7040 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1511 - val_sparse_categorical_accuracy: 0.5721
Epoch 1770/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7109 - sparse_categorical_accuracy: 0.6861 - val_loss: 1.1563 - val_sparse_categorical_accuracy: 0.5721
Epoch 1771/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7056 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1811 - val_sparse_categorical_accuracy: 0.5590
Epoch 1772/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7102 - sparse_categorical_accuracy: 0.6818 - val_loss: 1.1975 - val_sparse_categorical_accuracy: 0.5677
Epoch 1773/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7106 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1776 - val_sparse_categorical_accuracy: 0.5721
Epoch 1774/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7080 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1630 - val_sparse_categorical_accuracy: 0.5633
Epoch 1775/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7107 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1598 - val_sparse_categorical_accuracy: 0.5677
Epoch 1776/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7039 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1800 - val_sparse_categorical_accuracy: 0.5590
Epoch 1777/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7084 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1722 - val_sparse_categorical_accuracy: 0.5721
Epoch 1778/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7020 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1759 - val_sparse_categorical_accuracy: 0.5721
Epoch 1779/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7112 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1710 - val_sparse_categorical_accuracy: 0.5808
Epoch 1780/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7034 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1716 - val_sparse_categorical_accuracy: 0.5633
Epoch 1781/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7069 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1655 - val_sparse_categorical_accuracy: 0.5764
Epoch 1782/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7049 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1681 - val_sparse_categorical_accuracy: 0.5502
Epoch 1783/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7060 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1739 - val_sparse_categorical_accuracy: 0.5764
Epoch 1784/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7034 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1693 - val_sparse_categorical_accuracy: 0.5721
Epoch 1785/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7051 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1616 - val_sparse_categorical_accuracy: 0.5677
Epoch 1786/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7058 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1578 - val_sparse_categorical_accuracy: 0.5590
Epoch 1787/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7024 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1469 - val_sparse_categorical_accuracy: 0.5721
Epoch 1788/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7045 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1607 - val_sparse_categorical_accuracy: 0.5677
Epoch 1789/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7001 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1946 - val_sparse_categorical_accuracy: 0.5633
Epoch 1790/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7053 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1902 - val_sparse_categorical_accuracy: 0.5764
Epoch 1791/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7050 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1656 - val_sparse_categorical_accuracy: 0.5764
Epoch 1792/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7080 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1658 - val_sparse_categorical_accuracy: 0.5633
Epoch 1793/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7068 - sparse_categorical_accuracy: 0.6876 - val_loss: 1.1773 - val_sparse_categorical_accuracy: 0.5677
Epoch 1794/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7039 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1784 - val_sparse_categorical_accuracy: 0.5721
Epoch 1795/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7014 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1660 - val_sparse_categorical_accuracy: 0.5721
Epoch 1796/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7071 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1666 - val_sparse_categorical_accuracy: 0.5721
Epoch 1797/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7044 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1870 - val_sparse_categorical_accuracy: 0.5415
Epoch 1798/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7073 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1819 - val_sparse_categorical_accuracy: 0.5546
Epoch 1799/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7062 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1764 - val_sparse_categorical_accuracy: 0.5721
Epoch 1800/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7113 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1812 - val_sparse_categorical_accuracy: 0.5677
Epoch 1801/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7049 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1828 - val_sparse_categorical_accuracy: 0.5764
Epoch 1802/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7034 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1668 - val_sparse_categorical_accuracy: 0.5808
Epoch 1803/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7055 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.1514 - val_sparse_categorical_accuracy: 0.5939
Epoch 1804/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7024 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1621 - val_sparse_categorical_accuracy: 0.5764
Epoch 1805/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7015 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1735 - val_sparse_categorical_accuracy: 0.5590
Epoch 1806/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7016 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1826 - val_sparse_categorical_accuracy: 0.5459
Epoch 1807/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7042 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1704 - val_sparse_categorical_accuracy: 0.5721
Epoch 1808/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7041 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1600 - val_sparse_categorical_accuracy: 0.5852
Epoch 1809/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7045 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.1661 - val_sparse_categorical_accuracy: 0.5764
Epoch 1810/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7016 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1735 - val_sparse_categorical_accuracy: 0.5764
Epoch 1811/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7043 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1695 - val_sparse_categorical_accuracy: 0.5721
Epoch 1812/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7042 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1591 - val_sparse_categorical_accuracy: 0.5764
Epoch 1813/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7011 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1644 - val_sparse_categorical_accuracy: 0.5633
Epoch 1814/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7015 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1694 - val_sparse_categorical_accuracy: 0.5633
Epoch 1815/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7045 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1693 - val_sparse_categorical_accuracy: 0.5677
Epoch 1816/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7027 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1735 - val_sparse_categorical_accuracy: 0.5633
Epoch 1817/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6999 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1825 - val_sparse_categorical_accuracy: 0.5764
Epoch 1818/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7034 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1906 - val_sparse_categorical_accuracy: 0.5677
Epoch 1819/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7011 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1762 - val_sparse_categorical_accuracy: 0.5502
Epoch 1820/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7066 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1646 - val_sparse_categorical_accuracy: 0.5764
Epoch 1821/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7017 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1740 - val_sparse_categorical_accuracy: 0.5677
Epoch 1822/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7086 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1864 - val_sparse_categorical_accuracy: 0.5852
Epoch 1823/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7024 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1861 - val_sparse_categorical_accuracy: 0.5502
Epoch 1824/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7118 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1489 - val_sparse_categorical_accuracy: 0.5721
Epoch 1825/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7062 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1610 - val_sparse_categorical_accuracy: 0.5633
Epoch 1826/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7015 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1881 - val_sparse_categorical_accuracy: 0.5590
Epoch 1827/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7001 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1901 - val_sparse_categorical_accuracy: 0.5677
Epoch 1828/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6986 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1800 - val_sparse_categorical_accuracy: 0.5633
Epoch 1829/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6993 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1695 - val_sparse_categorical_accuracy: 0.5502
Epoch 1830/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7012 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1739 - val_sparse_categorical_accuracy: 0.5459
Epoch 1831/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7003 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1794 - val_sparse_categorical_accuracy: 0.5546
Epoch 1832/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6985 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1781 - val_sparse_categorical_accuracy: 0.5502
Epoch 1833/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7045 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1775 - val_sparse_categorical_accuracy: 0.5546
Epoch 1834/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7029 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1893 - val_sparse_categorical_accuracy: 0.5633
Epoch 1835/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.7016 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1855 - val_sparse_categorical_accuracy: 0.5721
Epoch 1836/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7001 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1707 - val_sparse_categorical_accuracy: 0.5721
Epoch 1837/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7010 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.1803 - val_sparse_categorical_accuracy: 0.5677
Epoch 1838/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7000 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.1903 - val_sparse_categorical_accuracy: 0.5590
Epoch 1839/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7033 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1756 - val_sparse_categorical_accuracy: 0.5764
Epoch 1840/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7014 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1588 - val_sparse_categorical_accuracy: 0.5633
Epoch 1841/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7036 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1682 - val_sparse_categorical_accuracy: 0.5633
Epoch 1842/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7044 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1776 - val_sparse_categorical_accuracy: 0.5633
Epoch 1843/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7002 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1818 - val_sparse_categorical_accuracy: 0.5677
Epoch 1844/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7038 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1831 - val_sparse_categorical_accuracy: 0.5764
Epoch 1845/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7008 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1727 - val_sparse_categorical_accuracy: 0.5764
Epoch 1846/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.7010 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1744 - val_sparse_categorical_accuracy: 0.5939
Epoch 1847/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7017 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1776 - val_sparse_categorical_accuracy: 0.5633
Epoch 1848/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7025 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1817 - val_sparse_categorical_accuracy: 0.5721
Epoch 1849/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7040 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1956 - val_sparse_categorical_accuracy: 0.5502
Epoch 1850/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6993 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1789 - val_sparse_categorical_accuracy: 0.5590
Epoch 1851/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6981 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1769 - val_sparse_categorical_accuracy: 0.5764
Epoch 1852/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7013 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1792 - val_sparse_categorical_accuracy: 0.5633
Epoch 1853/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6997 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.1914 - val_sparse_categorical_accuracy: 0.5415
Epoch 1854/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7014 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.2004 - val_sparse_categorical_accuracy: 0.5590
Epoch 1855/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7041 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.2076 - val_sparse_categorical_accuracy: 0.5502
Epoch 1856/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7077 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1847 - val_sparse_categorical_accuracy: 0.5808
Epoch 1857/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7015 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1552 - val_sparse_categorical_accuracy: 0.5939
Epoch 1858/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7033 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1660 - val_sparse_categorical_accuracy: 0.5983
Epoch 1859/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7128 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1797 - val_sparse_categorical_accuracy: 0.5721
Epoch 1860/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7007 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2129 - val_sparse_categorical_accuracy: 0.5415
Epoch 1861/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7065 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1995 - val_sparse_categorical_accuracy: 0.5502
Epoch 1862/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7082 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1810 - val_sparse_categorical_accuracy: 0.5633
Epoch 1863/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7031 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1775 - val_sparse_categorical_accuracy: 0.5590
Epoch 1864/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7028 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1973 - val_sparse_categorical_accuracy: 0.5590
Epoch 1865/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7099 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1896 - val_sparse_categorical_accuracy: 0.5721
Epoch 1866/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6995 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1743 - val_sparse_categorical_accuracy: 0.5546
Epoch 1867/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7024 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2006 - val_sparse_categorical_accuracy: 0.5546
Epoch 1868/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7092 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2196 - val_sparse_categorical_accuracy: 0.5415
Epoch 1869/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7009 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1900 - val_sparse_categorical_accuracy: 0.5808
Epoch 1870/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7062 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1746 - val_sparse_categorical_accuracy: 0.5983
Epoch 1871/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7015 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1875 - val_sparse_categorical_accuracy: 0.5240
Epoch 1872/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7052 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1846 - val_sparse_categorical_accuracy: 0.5590
Epoch 1873/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7027 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1733 - val_sparse_categorical_accuracy: 0.5546
Epoch 1874/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7111 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1702 - val_sparse_categorical_accuracy: 0.5808
Epoch 1875/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7014 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.2085 - val_sparse_categorical_accuracy: 0.5284
Epoch 1876/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7075 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1924 - val_sparse_categorical_accuracy: 0.5764
Epoch 1877/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6991 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.1833 - val_sparse_categorical_accuracy: 0.5808
Epoch 1878/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7031 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.2085 - val_sparse_categorical_accuracy: 0.5633
Epoch 1879/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7013 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2401 - val_sparse_categorical_accuracy: 0.5459
Epoch 1880/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7094 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2073 - val_sparse_categorical_accuracy: 0.5546
Epoch 1881/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6942 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1856 - val_sparse_categorical_accuracy: 0.5764
Epoch 1882/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7145 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1894 - val_sparse_categorical_accuracy: 0.6026
Epoch 1883/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7030 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.2240 - val_sparse_categorical_accuracy: 0.5109
Epoch 1884/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7175 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1756 - val_sparse_categorical_accuracy: 0.5764
Epoch 1885/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7015 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1692 - val_sparse_categorical_accuracy: 0.5677
Epoch 1886/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7085 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1840 - val_sparse_categorical_accuracy: 0.5677
Epoch 1887/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7005 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.2212 - val_sparse_categorical_accuracy: 0.5197
Epoch 1888/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7084 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1947 - val_sparse_categorical_accuracy: 0.5677
Epoch 1889/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7012 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1887 - val_sparse_categorical_accuracy: 0.5502
Epoch 1890/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7152 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1839 - val_sparse_categorical_accuracy: 0.5590
Epoch 1891/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6955 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2096 - val_sparse_categorical_accuracy: 0.5153
Epoch 1892/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7144 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.2001 - val_sparse_categorical_accuracy: 0.5590
Epoch 1893/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7044 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1882 - val_sparse_categorical_accuracy: 0.5721
Epoch 1894/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7052 - sparse_categorical_accuracy: 0.6832 - val_loss: 1.1997 - val_sparse_categorical_accuracy: 0.5721
Epoch 1895/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7041 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.2080 - val_sparse_categorical_accuracy: 0.5633
Epoch 1896/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7000 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1793 - val_sparse_categorical_accuracy: 0.5677
Epoch 1897/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7042 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1658 - val_sparse_categorical_accuracy: 0.5764
Epoch 1898/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6998 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.1852 - val_sparse_categorical_accuracy: 0.5852
Epoch 1899/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7004 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1996 - val_sparse_categorical_accuracy: 0.5677
Epoch 1900/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7025 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1624 - val_sparse_categorical_accuracy: 0.5764
Epoch 1901/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7010 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.1536 - val_sparse_categorical_accuracy: 0.5808
Epoch 1902/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7011 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1825 - val_sparse_categorical_accuracy: 0.5415
Epoch 1903/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7030 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2012 - val_sparse_categorical_accuracy: 0.5721
Epoch 1904/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7021 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2032 - val_sparse_categorical_accuracy: 0.5546
Epoch 1905/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7047 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1883 - val_sparse_categorical_accuracy: 0.5590
Epoch 1906/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6997 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1902 - val_sparse_categorical_accuracy: 0.5328
Epoch 1907/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7068 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1688 - val_sparse_categorical_accuracy: 0.5939
Epoch 1908/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7005 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1653 - val_sparse_categorical_accuracy: 0.5895
Epoch 1909/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6998 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1898 - val_sparse_categorical_accuracy: 0.5895
Epoch 1910/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6997 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2018 - val_sparse_categorical_accuracy: 0.5721
Epoch 1911/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6957 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1951 - val_sparse_categorical_accuracy: 0.5633
Epoch 1912/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6972 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1763 - val_sparse_categorical_accuracy: 0.5633
Epoch 1913/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6971 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1764 - val_sparse_categorical_accuracy: 0.5808
Epoch 1914/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7008 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.2006 - val_sparse_categorical_accuracy: 0.5764
Epoch 1915/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6953 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2138 - val_sparse_categorical_accuracy: 0.5502
Epoch 1916/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7006 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2005 - val_sparse_categorical_accuracy: 0.5590
Epoch 1917/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6958 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1919 - val_sparse_categorical_accuracy: 0.5677
Epoch 1918/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6987 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1785 - val_sparse_categorical_accuracy: 0.5764
Epoch 1919/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6965 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.2006 - val_sparse_categorical_accuracy: 0.5721
Epoch 1920/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6964 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2181 - val_sparse_categorical_accuracy: 0.5502
Epoch 1921/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6960 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1851 - val_sparse_categorical_accuracy: 0.5721
Epoch 1922/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6951 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1780 - val_sparse_categorical_accuracy: 0.5764
Epoch 1923/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6985 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2071 - val_sparse_categorical_accuracy: 0.5808
Epoch 1924/2000
2/2 [==============================] - 0s 20ms/step - loss: 0.6978 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2114 - val_sparse_categorical_accuracy: 0.5546
Epoch 1925/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6926 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1931 - val_sparse_categorical_accuracy: 0.5983
Epoch 1926/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7038 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1822 - val_sparse_categorical_accuracy: 0.5852
Epoch 1927/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.7042 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1851 - val_sparse_categorical_accuracy: 0.5459
Epoch 1928/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6961 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1889 - val_sparse_categorical_accuracy: 0.5677
Epoch 1929/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6961 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.2019 - val_sparse_categorical_accuracy: 0.5546
Epoch 1930/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.7015 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1993 - val_sparse_categorical_accuracy: 0.5764
Epoch 1931/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6961 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2078 - val_sparse_categorical_accuracy: 0.5633
Epoch 1932/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6948 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1997 - val_sparse_categorical_accuracy: 0.5852
Epoch 1933/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6968 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.1886 - val_sparse_categorical_accuracy: 0.5764
Epoch 1934/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6937 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1990 - val_sparse_categorical_accuracy: 0.5721
Epoch 1935/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6921 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2047 - val_sparse_categorical_accuracy: 0.5546
Epoch 1936/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6968 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1818 - val_sparse_categorical_accuracy: 0.5721
Epoch 1937/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6996 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1698 - val_sparse_categorical_accuracy: 0.5590
Epoch 1938/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6963 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1907 - val_sparse_categorical_accuracy: 0.5633
Epoch 1939/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6941 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2088 - val_sparse_categorical_accuracy: 0.5677
Epoch 1940/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6969 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1977 - val_sparse_categorical_accuracy: 0.5721
Epoch 1941/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6923 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1734 - val_sparse_categorical_accuracy: 0.5808
Epoch 1942/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6955 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1633 - val_sparse_categorical_accuracy: 0.5852
Epoch 1943/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6970 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1660 - val_sparse_categorical_accuracy: 0.5852
Epoch 1944/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6962 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1871 - val_sparse_categorical_accuracy: 0.5721
Epoch 1945/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6921 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.2043 - val_sparse_categorical_accuracy: 0.5721
Epoch 1946/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6948 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2062 - val_sparse_categorical_accuracy: 0.5459
Epoch 1947/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6958 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1970 - val_sparse_categorical_accuracy: 0.5502
Epoch 1948/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6912 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1963 - val_sparse_categorical_accuracy: 0.5415
Epoch 1949/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6973 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1955 - val_sparse_categorical_accuracy: 0.5721
Epoch 1950/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6965 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2060 - val_sparse_categorical_accuracy: 0.5721
Epoch 1951/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6973 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2049 - val_sparse_categorical_accuracy: 0.5677
Epoch 1952/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6922 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2071 - val_sparse_categorical_accuracy: 0.5677
Epoch 1953/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6919 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2241 - val_sparse_categorical_accuracy: 0.5633
Epoch 1954/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6926 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2291 - val_sparse_categorical_accuracy: 0.5546
Epoch 1955/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6939 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2127 - val_sparse_categorical_accuracy: 0.5590
Epoch 1956/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6910 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.1927 - val_sparse_categorical_accuracy: 0.5721
Epoch 1957/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6939 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1880 - val_sparse_categorical_accuracy: 0.5721
Epoch 1958/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6937 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1939 - val_sparse_categorical_accuracy: 0.5852
Epoch 1959/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6944 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1933 - val_sparse_categorical_accuracy: 0.5852
Epoch 1960/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6970 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.1846 - val_sparse_categorical_accuracy: 0.5764
Epoch 1961/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6953 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1920 - val_sparse_categorical_accuracy: 0.5808
Epoch 1962/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6944 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2137 - val_sparse_categorical_accuracy: 0.5764
Epoch 1963/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6966 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2051 - val_sparse_categorical_accuracy: 0.5546
Epoch 1964/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6957 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.1628 - val_sparse_categorical_accuracy: 0.5721
Epoch 1965/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6951 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1607 - val_sparse_categorical_accuracy: 0.5808
Epoch 1966/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6944 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1990 - val_sparse_categorical_accuracy: 0.5459
Epoch 1967/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6953 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2053 - val_sparse_categorical_accuracy: 0.5633
Epoch 1968/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6951 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1902 - val_sparse_categorical_accuracy: 0.5721
Epoch 1969/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6957 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2096 - val_sparse_categorical_accuracy: 0.5502
Epoch 1970/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6920 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2097 - val_sparse_categorical_accuracy: 0.5546
Epoch 1971/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6921 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.1825 - val_sparse_categorical_accuracy: 0.5764
Epoch 1972/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6919 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1953 - val_sparse_categorical_accuracy: 0.5764
Epoch 1973/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6904 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2242 - val_sparse_categorical_accuracy: 0.5590
Epoch 1974/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6950 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2102 - val_sparse_categorical_accuracy: 0.5590
Epoch 1975/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6930 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1945 - val_sparse_categorical_accuracy: 0.5590
Epoch 1976/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6980 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2121 - val_sparse_categorical_accuracy: 0.5546
Epoch 1977/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6920 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2187 - val_sparse_categorical_accuracy: 0.5415
Epoch 1978/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6899 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.1945 - val_sparse_categorical_accuracy: 0.5721
Epoch 1979/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6920 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1698 - val_sparse_categorical_accuracy: 0.5895
Epoch 1980/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6933 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.1833 - val_sparse_categorical_accuracy: 0.5721
Epoch 1981/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6950 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2014 - val_sparse_categorical_accuracy: 0.5590
Epoch 1982/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6899 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1837 - val_sparse_categorical_accuracy: 0.5590
Epoch 1983/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6962 - sparse_categorical_accuracy: 0.6905 - val_loss: 1.1881 - val_sparse_categorical_accuracy: 0.5546
Epoch 1984/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6955 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.2220 - val_sparse_categorical_accuracy: 0.5328
Epoch 1985/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6951 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2244 - val_sparse_categorical_accuracy: 0.5415
Epoch 1986/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6930 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.1973 - val_sparse_categorical_accuracy: 0.5895
Epoch 1987/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6924 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.1890 - val_sparse_categorical_accuracy: 0.5677
Epoch 1988/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6914 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2153 - val_sparse_categorical_accuracy: 0.5415
Epoch 1989/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6919 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.2029 - val_sparse_categorical_accuracy: 0.5502
Epoch 1990/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6883 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.1917 - val_sparse_categorical_accuracy: 0.5677
Epoch 1991/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6899 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2024 - val_sparse_categorical_accuracy: 0.5721
Epoch 1992/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6874 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2246 - val_sparse_categorical_accuracy: 0.5546
Epoch 1993/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6921 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2099 - val_sparse_categorical_accuracy: 0.5590
Epoch 1994/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6894 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.1943 - val_sparse_categorical_accuracy: 0.5633
Epoch 1995/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6946 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2065 - val_sparse_categorical_accuracy: 0.5808
Epoch 1996/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6918 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2255 - val_sparse_categorical_accuracy: 0.5546
Epoch 1997/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6953 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2034 - val_sparse_categorical_accuracy: 0.5764
Epoch 1998/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6906 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.1887 - val_sparse_categorical_accuracy: 0.5721
Epoch 1999/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6946 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.1949 - val_sparse_categorical_accuracy: 0.5721
Epoch 2000/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6912 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2009 - val_sparse_categorical_accuracy: 0.5415
</pre>
<pre>
<keras.callbacks.History at 0x7f7b7948aee0>
</pre>
## How to save Model based on Epochs and Metrics


Set path and type of metrics

- modelpath = '../{epoch:02d}-{val_loss:.4f}.hdf5'



```python
modelpath = '../{epoch:02d}-{val_loss:.4f}.hdf5'
```


```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath = modelpath, verbose = 1)
```


```python
history = model.fit(X_tn,y_tn, validation_split = 0.25, epochs = 100, batch_size = 500, callbacks = [checkpointer],verbose = 0)
```

<pre>

Epoch 1: saving model to ../01-1.2055.hdf5

Epoch 2: saving model to ../02-1.1953.hdf5

Epoch 3: saving model to ../03-1.1920.hdf5

Epoch 4: saving model to ../04-1.2128.hdf5

Epoch 5: saving model to ../05-1.2236.hdf5

Epoch 6: saving model to ../06-1.1983.hdf5

Epoch 7: saving model to ../07-1.1833.hdf5

Epoch 8: saving model to ../08-1.1979.hdf5

Epoch 9: saving model to ../09-1.2169.hdf5

Epoch 10: saving model to ../10-1.2091.hdf5

Epoch 11: saving model to ../11-1.2003.hdf5

Epoch 12: saving model to ../12-1.2214.hdf5

Epoch 13: saving model to ../13-1.2338.hdf5

Epoch 14: saving model to ../14-1.2111.hdf5

Epoch 15: saving model to ../15-1.2009.hdf5

Epoch 16: saving model to ../16-1.1964.hdf5

Epoch 17: saving model to ../17-1.1944.hdf5

Epoch 18: saving model to ../18-1.1982.hdf5

Epoch 19: saving model to ../19-1.2159.hdf5

Epoch 20: saving model to ../20-1.2244.hdf5

Epoch 21: saving model to ../21-1.2064.hdf5

Epoch 22: saving model to ../22-1.2026.hdf5

Epoch 23: saving model to ../23-1.2123.hdf5

Epoch 24: saving model to ../24-1.2083.hdf5

Epoch 25: saving model to ../25-1.2128.hdf5

Epoch 26: saving model to ../26-1.2155.hdf5

Epoch 27: saving model to ../27-1.2208.hdf5

Epoch 28: saving model to ../28-1.2011.hdf5

Epoch 29: saving model to ../29-1.1907.hdf5

Epoch 30: saving model to ../30-1.2125.hdf5

Epoch 31: saving model to ../31-1.2132.hdf5

Epoch 32: saving model to ../32-1.2145.hdf5

Epoch 33: saving model to ../33-1.2099.hdf5

Epoch 34: saving model to ../34-1.2156.hdf5

Epoch 35: saving model to ../35-1.2146.hdf5

Epoch 36: saving model to ../36-1.2050.hdf5

Epoch 37: saving model to ../37-1.1920.hdf5

Epoch 38: saving model to ../38-1.1852.hdf5

Epoch 39: saving model to ../39-1.1952.hdf5

Epoch 40: saving model to ../40-1.2076.hdf5

Epoch 41: saving model to ../41-1.1985.hdf5

Epoch 42: saving model to ../42-1.2092.hdf5

Epoch 43: saving model to ../43-1.2205.hdf5

Epoch 44: saving model to ../44-1.1971.hdf5

Epoch 45: saving model to ../45-1.1913.hdf5

Epoch 46: saving model to ../46-1.2274.hdf5

Epoch 47: saving model to ../47-1.2295.hdf5

Epoch 48: saving model to ../48-1.1995.hdf5

Epoch 49: saving model to ../49-1.2073.hdf5

Epoch 50: saving model to ../50-1.2358.hdf5

Epoch 51: saving model to ../51-1.2017.hdf5

Epoch 52: saving model to ../52-1.1900.hdf5

Epoch 53: saving model to ../53-1.2095.hdf5

Epoch 54: saving model to ../54-1.2188.hdf5

Epoch 55: saving model to ../55-1.1750.hdf5

Epoch 56: saving model to ../56-1.1699.hdf5

Epoch 57: saving model to ../57-1.2114.hdf5

Epoch 58: saving model to ../58-1.2604.hdf5

Epoch 59: saving model to ../59-1.2117.hdf5

Epoch 60: saving model to ../60-1.1818.hdf5

Epoch 61: saving model to ../61-1.2116.hdf5

Epoch 62: saving model to ../62-1.2527.hdf5

Epoch 63: saving model to ../63-1.2301.hdf5

Epoch 64: saving model to ../64-1.1812.hdf5

Epoch 65: saving model to ../65-1.1828.hdf5

Epoch 66: saving model to ../66-1.1979.hdf5

Epoch 67: saving model to ../67-1.2194.hdf5

Epoch 68: saving model to ../68-1.2092.hdf5

Epoch 69: saving model to ../69-1.2120.hdf5

Epoch 70: saving model to ../70-1.2228.hdf5

Epoch 71: saving model to ../71-1.2178.hdf5

Epoch 72: saving model to ../72-1.2079.hdf5

Epoch 73: saving model to ../73-1.2166.hdf5

Epoch 74: saving model to ../74-1.2076.hdf5

Epoch 75: saving model to ../75-1.2026.hdf5

Epoch 76: saving model to ../76-1.2268.hdf5

Epoch 77: saving model to ../77-1.2396.hdf5

Epoch 78: saving model to ../78-1.2107.hdf5

Epoch 79: saving model to ../79-1.2055.hdf5

Epoch 80: saving model to ../80-1.2338.hdf5

Epoch 81: saving model to ../81-1.2309.hdf5

Epoch 82: saving model to ../82-1.2116.hdf5

Epoch 83: saving model to ../83-1.2128.hdf5

Epoch 84: saving model to ../84-1.2152.hdf5

Epoch 85: saving model to ../85-1.2035.hdf5

Epoch 86: saving model to ../86-1.2121.hdf5

Epoch 87: saving model to ../87-1.2156.hdf5

Epoch 88: saving model to ../88-1.2077.hdf5

Epoch 89: saving model to ../89-1.2050.hdf5

Epoch 90: saving model to ../90-1.2227.hdf5

Epoch 91: saving model to ../91-1.2332.hdf5

Epoch 92: saving model to ../92-1.2121.hdf5

Epoch 93: saving model to ../93-1.2161.hdf5

Epoch 94: saving model to ../94-1.2187.hdf5

Epoch 95: saving model to ../95-1.2066.hdf5

Epoch 96: saving model to ../96-1.2071.hdf5

Epoch 97: saving model to ../97-1.2196.hdf5

Epoch 98: saving model to ../98-1.2253.hdf5

Epoch 99: saving model to ../99-1.2315.hdf5

Epoch 100: saving model to ../100-1.2262.hdf5
</pre>

```python
hist_df = pd.DataFrame(history.history)
hist_df
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
      <th>loss</th>
      <th>sparse_categorical_accuracy</th>
      <th>val_loss</th>
      <th>val_sparse_categorical_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.688872</td>
      <td>0.700730</td>
      <td>1.205545</td>
      <td>0.563319</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.692343</td>
      <td>0.699270</td>
      <td>1.195296</td>
      <td>0.563319</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.690377</td>
      <td>0.699270</td>
      <td>1.192016</td>
      <td>0.572052</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.688359</td>
      <td>0.703650</td>
      <td>1.212778</td>
      <td>0.554585</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.691166</td>
      <td>0.705109</td>
      <td>1.223635</td>
      <td>0.558952</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.685724</td>
      <td>0.708029</td>
      <td>1.207067</td>
      <td>0.585153</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.684251</td>
      <td>0.710949</td>
      <td>1.219643</td>
      <td>0.567686</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.683109</td>
      <td>0.705109</td>
      <td>1.225324</td>
      <td>0.563319</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.683927</td>
      <td>0.706569</td>
      <td>1.231506</td>
      <td>0.558952</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.682556</td>
      <td>0.712409</td>
      <td>1.226198</td>
      <td>0.558952</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>



```python
history.params
```

<pre>
{'verbose': 0, 'epochs': 100, 'steps': 2}
</pre>

```python
np.array(history.epoch).flatten()
```

<pre>
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
</pre>
## Visualize Overfitting



```python
history = model.fit(X_tn, y_tn, epochs = 2000, batch_size = 500, validation_split = 0.25)
```

<pre>
Epoch 1/2000
2/2 [==============================] - 0s 43ms/step - loss: 0.6840 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2237 - val_sparse_categorical_accuracy: 0.5677
Epoch 2/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6810 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2296 - val_sparse_categorical_accuracy: 0.5677
Epoch 3/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6859 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2173 - val_sparse_categorical_accuracy: 0.5633
Epoch 4/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6842 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2148 - val_sparse_categorical_accuracy: 0.5633
Epoch 5/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6860 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2216 - val_sparse_categorical_accuracy: 0.5502
Epoch 6/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6837 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2237 - val_sparse_categorical_accuracy: 0.5633
Epoch 7/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6863 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2230 - val_sparse_categorical_accuracy: 0.5677
Epoch 8/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6852 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2169 - val_sparse_categorical_accuracy: 0.5633
Epoch 9/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6859 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2385 - val_sparse_categorical_accuracy: 0.5459
Epoch 10/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6869 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2204 - val_sparse_categorical_accuracy: 0.5721
Epoch 11/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6875 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2132 - val_sparse_categorical_accuracy: 0.5633
Epoch 12/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6892 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2337 - val_sparse_categorical_accuracy: 0.5502
Epoch 13/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6865 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2216 - val_sparse_categorical_accuracy: 0.5590
Epoch 14/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6826 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.1928 - val_sparse_categorical_accuracy: 0.5895
Epoch 15/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6900 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2034 - val_sparse_categorical_accuracy: 0.5808
Epoch 16/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6854 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2289 - val_sparse_categorical_accuracy: 0.5590
Epoch 17/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6863 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1907 - val_sparse_categorical_accuracy: 0.5633
Epoch 18/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6882 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.1801 - val_sparse_categorical_accuracy: 0.5764
Epoch 19/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6925 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2314 - val_sparse_categorical_accuracy: 0.5415
Epoch 20/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6873 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2721 - val_sparse_categorical_accuracy: 0.5502
Epoch 21/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6916 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.2285 - val_sparse_categorical_accuracy: 0.5502
Epoch 22/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6848 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2044 - val_sparse_categorical_accuracy: 0.5502
Epoch 23/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6944 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2199 - val_sparse_categorical_accuracy: 0.5284
Epoch 24/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6893 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2452 - val_sparse_categorical_accuracy: 0.5590
Epoch 25/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6870 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2239 - val_sparse_categorical_accuracy: 0.5895
Epoch 26/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6859 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.1946 - val_sparse_categorical_accuracy: 0.5721
Epoch 27/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6888 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2262 - val_sparse_categorical_accuracy: 0.5459
Epoch 28/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6975 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2267 - val_sparse_categorical_accuracy: 0.5677
Epoch 29/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6865 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2063 - val_sparse_categorical_accuracy: 0.5459
Epoch 30/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6999 - sparse_categorical_accuracy: 0.6920 - val_loss: 1.2068 - val_sparse_categorical_accuracy: 0.5764
Epoch 31/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6899 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2490 - val_sparse_categorical_accuracy: 0.5284
Epoch 32/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6925 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2350 - val_sparse_categorical_accuracy: 0.5721
Epoch 33/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6876 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2255 - val_sparse_categorical_accuracy: 0.5677
Epoch 34/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6863 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2483 - val_sparse_categorical_accuracy: 0.5633
Epoch 35/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6921 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2415 - val_sparse_categorical_accuracy: 0.5502
Epoch 36/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6833 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2211 - val_sparse_categorical_accuracy: 0.5939
Epoch 37/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6871 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2180 - val_sparse_categorical_accuracy: 0.5939
Epoch 38/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6887 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2335 - val_sparse_categorical_accuracy: 0.5546
Epoch 39/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6832 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2695 - val_sparse_categorical_accuracy: 0.5328
Epoch 40/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6869 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2472 - val_sparse_categorical_accuracy: 0.5633
Epoch 41/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6833 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2238 - val_sparse_categorical_accuracy: 0.5633
Epoch 42/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6844 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2351 - val_sparse_categorical_accuracy: 0.5633
Epoch 43/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6864 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2361 - val_sparse_categorical_accuracy: 0.5546
Epoch 44/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6806 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2241 - val_sparse_categorical_accuracy: 0.5633
Epoch 45/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6913 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2253 - val_sparse_categorical_accuracy: 0.5764
Epoch 46/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6870 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2379 - val_sparse_categorical_accuracy: 0.5371
Epoch 47/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6892 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2137 - val_sparse_categorical_accuracy: 0.5764
Epoch 48/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6817 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2318 - val_sparse_categorical_accuracy: 0.5764
Epoch 49/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6866 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2507 - val_sparse_categorical_accuracy: 0.5459
Epoch 50/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6883 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2731 - val_sparse_categorical_accuracy: 0.5328
Epoch 51/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6938 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2285 - val_sparse_categorical_accuracy: 0.5633
Epoch 52/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6830 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2098 - val_sparse_categorical_accuracy: 0.5677
Epoch 53/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6842 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2281 - val_sparse_categorical_accuracy: 0.5721
Epoch 54/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6869 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2483 - val_sparse_categorical_accuracy: 0.5415
Epoch 55/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6875 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2211 - val_sparse_categorical_accuracy: 0.5633
Epoch 56/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6848 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2230 - val_sparse_categorical_accuracy: 0.5590
Epoch 57/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6844 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2596 - val_sparse_categorical_accuracy: 0.5546
Epoch 58/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6837 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2647 - val_sparse_categorical_accuracy: 0.5459
Epoch 59/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6816 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2232 - val_sparse_categorical_accuracy: 0.5546
Epoch 60/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6858 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.2179 - val_sparse_categorical_accuracy: 0.5721
Epoch 61/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6840 - sparse_categorical_accuracy: 0.6934 - val_loss: 1.2522 - val_sparse_categorical_accuracy: 0.5328
Epoch 62/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6825 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2561 - val_sparse_categorical_accuracy: 0.5764
Epoch 63/2000
2/2 [==============================] - 0s 21ms/step - loss: 0.6873 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2317 - val_sparse_categorical_accuracy: 0.5590
Epoch 64/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6840 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2235 - val_sparse_categorical_accuracy: 0.5459
Epoch 65/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6847 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2539 - val_sparse_categorical_accuracy: 0.5459
Epoch 66/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6866 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2478 - val_sparse_categorical_accuracy: 0.5764
Epoch 67/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6875 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.1962 - val_sparse_categorical_accuracy: 0.5590
Epoch 68/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6889 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.1890 - val_sparse_categorical_accuracy: 0.5633
Epoch 69/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6895 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2298 - val_sparse_categorical_accuracy: 0.5371
Epoch 70/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6890 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2515 - val_sparse_categorical_accuracy: 0.5633
Epoch 71/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6940 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2177 - val_sparse_categorical_accuracy: 0.5633
Epoch 72/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6869 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.1876 - val_sparse_categorical_accuracy: 0.5677
Epoch 73/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6995 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.1839 - val_sparse_categorical_accuracy: 0.5633
Epoch 74/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6963 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2136 - val_sparse_categorical_accuracy: 0.5590
Epoch 75/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6945 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2538 - val_sparse_categorical_accuracy: 0.5459
Epoch 76/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6942 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2712 - val_sparse_categorical_accuracy: 0.5502
Epoch 77/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6931 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2567 - val_sparse_categorical_accuracy: 0.5590
Epoch 78/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6854 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2282 - val_sparse_categorical_accuracy: 0.5590
Epoch 79/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6827 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2147 - val_sparse_categorical_accuracy: 0.5808
Epoch 80/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6851 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2142 - val_sparse_categorical_accuracy: 0.5852
Epoch 81/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6881 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2297 - val_sparse_categorical_accuracy: 0.5415
Epoch 82/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6870 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2226 - val_sparse_categorical_accuracy: 0.5677
Epoch 83/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6852 - sparse_categorical_accuracy: 0.6891 - val_loss: 1.2273 - val_sparse_categorical_accuracy: 0.5677
Epoch 84/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6821 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2459 - val_sparse_categorical_accuracy: 0.5284
Epoch 85/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6852 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2477 - val_sparse_categorical_accuracy: 0.5546
Epoch 86/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6827 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2379 - val_sparse_categorical_accuracy: 0.5764
Epoch 87/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6805 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2259 - val_sparse_categorical_accuracy: 0.5677
Epoch 88/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6807 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2169 - val_sparse_categorical_accuracy: 0.5590
Epoch 89/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6837 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2186 - val_sparse_categorical_accuracy: 0.5546
Epoch 90/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6827 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2378 - val_sparse_categorical_accuracy: 0.5590
Epoch 91/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6793 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2621 - val_sparse_categorical_accuracy: 0.5633
Epoch 92/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6883 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2497 - val_sparse_categorical_accuracy: 0.5546
Epoch 93/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6817 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2519 - val_sparse_categorical_accuracy: 0.5328
Epoch 94/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6949 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2185 - val_sparse_categorical_accuracy: 0.5677
Epoch 95/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6851 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2340 - val_sparse_categorical_accuracy: 0.5808
Epoch 96/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6948 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.2380 - val_sparse_categorical_accuracy: 0.5677
Epoch 97/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6827 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2601 - val_sparse_categorical_accuracy: 0.5240
Epoch 98/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6982 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2322 - val_sparse_categorical_accuracy: 0.5721
Epoch 99/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6881 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2483 - val_sparse_categorical_accuracy: 0.5546
Epoch 100/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6881 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2543 - val_sparse_categorical_accuracy: 0.5459
Epoch 101/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6794 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2438 - val_sparse_categorical_accuracy: 0.5371
Epoch 102/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6918 - sparse_categorical_accuracy: 0.6949 - val_loss: 1.2057 - val_sparse_categorical_accuracy: 0.5764
Epoch 103/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6867 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2270 - val_sparse_categorical_accuracy: 0.6026
Epoch 104/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6877 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2374 - val_sparse_categorical_accuracy: 0.5721
Epoch 105/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6799 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2393 - val_sparse_categorical_accuracy: 0.5590
Epoch 106/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6858 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2462 - val_sparse_categorical_accuracy: 0.5546
Epoch 107/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6801 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2545 - val_sparse_categorical_accuracy: 0.5633
Epoch 108/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6793 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2497 - val_sparse_categorical_accuracy: 0.5633
Epoch 109/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6808 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2330 - val_sparse_categorical_accuracy: 0.5546
Epoch 110/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6777 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2288 - val_sparse_categorical_accuracy: 0.5721
Epoch 111/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6851 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2262 - val_sparse_categorical_accuracy: 0.5677
Epoch 112/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6814 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2364 - val_sparse_categorical_accuracy: 0.5415
Epoch 113/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6859 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2349 - val_sparse_categorical_accuracy: 0.5764
Epoch 114/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6781 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2400 - val_sparse_categorical_accuracy: 0.5677
Epoch 115/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6838 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2448 - val_sparse_categorical_accuracy: 0.5852
Epoch 116/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6826 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2578 - val_sparse_categorical_accuracy: 0.5502
Epoch 117/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6839 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2303 - val_sparse_categorical_accuracy: 0.5808
Epoch 118/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6767 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2295 - val_sparse_categorical_accuracy: 0.5764
Epoch 119/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6852 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2405 - val_sparse_categorical_accuracy: 0.5502
Epoch 120/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6778 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2660 - val_sparse_categorical_accuracy: 0.5240
Epoch 121/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6835 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2631 - val_sparse_categorical_accuracy: 0.5459
Epoch 122/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6799 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2362 - val_sparse_categorical_accuracy: 0.5721
Epoch 123/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6813 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2160 - val_sparse_categorical_accuracy: 0.5633
Epoch 124/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6828 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2447 - val_sparse_categorical_accuracy: 0.5415
Epoch 125/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6820 - sparse_categorical_accuracy: 0.6993 - val_loss: 1.2613 - val_sparse_categorical_accuracy: 0.5677
Epoch 126/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6790 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2324 - val_sparse_categorical_accuracy: 0.5677
Epoch 127/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6821 - sparse_categorical_accuracy: 0.7022 - val_loss: 1.2211 - val_sparse_categorical_accuracy: 0.5633
Epoch 128/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6814 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2566 - val_sparse_categorical_accuracy: 0.5459
Epoch 129/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6828 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2602 - val_sparse_categorical_accuracy: 0.5546
Epoch 130/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6800 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2189 - val_sparse_categorical_accuracy: 0.5764
Epoch 131/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6874 - sparse_categorical_accuracy: 0.6964 - val_loss: 1.2145 - val_sparse_categorical_accuracy: 0.5939
Epoch 132/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6884 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2574 - val_sparse_categorical_accuracy: 0.5415
Epoch 133/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6886 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2718 - val_sparse_categorical_accuracy: 0.5633
Epoch 134/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6797 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2500 - val_sparse_categorical_accuracy: 0.5764
Epoch 135/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6849 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2432 - val_sparse_categorical_accuracy: 0.5677
Epoch 136/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6824 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2561 - val_sparse_categorical_accuracy: 0.5459
Epoch 137/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6811 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2469 - val_sparse_categorical_accuracy: 0.5764
Epoch 138/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6762 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2375 - val_sparse_categorical_accuracy: 0.5764
Epoch 139/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6775 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2417 - val_sparse_categorical_accuracy: 0.5677
Epoch 140/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6773 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2641 - val_sparse_categorical_accuracy: 0.5328
Epoch 141/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6797 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2674 - val_sparse_categorical_accuracy: 0.5415
Epoch 142/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6770 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2563 - val_sparse_categorical_accuracy: 0.5546
Epoch 143/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6791 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2536 - val_sparse_categorical_accuracy: 0.5371
Epoch 144/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6777 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2538 - val_sparse_categorical_accuracy: 0.5459
Epoch 145/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6785 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2402 - val_sparse_categorical_accuracy: 0.5939
Epoch 146/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6790 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2507 - val_sparse_categorical_accuracy: 0.5721
Epoch 147/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6754 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2456 - val_sparse_categorical_accuracy: 0.5721
Epoch 148/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6787 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2355 - val_sparse_categorical_accuracy: 0.5808
Epoch 149/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6762 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2499 - val_sparse_categorical_accuracy: 0.5721
Epoch 150/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6819 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2393 - val_sparse_categorical_accuracy: 0.5895
Epoch 151/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6766 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2528 - val_sparse_categorical_accuracy: 0.5371
Epoch 152/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6797 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2827 - val_sparse_categorical_accuracy: 0.5371
Epoch 153/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6809 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2879 - val_sparse_categorical_accuracy: 0.5502
Epoch 154/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6811 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2520 - val_sparse_categorical_accuracy: 0.5371
Epoch 155/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6763 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2282 - val_sparse_categorical_accuracy: 0.5633
Epoch 156/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6796 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2407 - val_sparse_categorical_accuracy: 0.5677
Epoch 157/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6784 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2552 - val_sparse_categorical_accuracy: 0.5721
Epoch 158/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6799 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2248 - val_sparse_categorical_accuracy: 0.5983
Epoch 159/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6827 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2283 - val_sparse_categorical_accuracy: 0.5895
Epoch 160/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6791 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2683 - val_sparse_categorical_accuracy: 0.5546
Epoch 161/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6797 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2864 - val_sparse_categorical_accuracy: 0.5415
Epoch 162/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6812 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2637 - val_sparse_categorical_accuracy: 0.5721
Epoch 163/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6821 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2434 - val_sparse_categorical_accuracy: 0.6026
Epoch 164/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6791 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2490 - val_sparse_categorical_accuracy: 0.5284
Epoch 165/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6842 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2435 - val_sparse_categorical_accuracy: 0.5415
Epoch 166/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6777 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2449 - val_sparse_categorical_accuracy: 0.5764
Epoch 167/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6869 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2309 - val_sparse_categorical_accuracy: 0.5852
Epoch 168/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6825 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2606 - val_sparse_categorical_accuracy: 0.5284
Epoch 169/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6844 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2801 - val_sparse_categorical_accuracy: 0.5721
Epoch 170/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6824 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2726 - val_sparse_categorical_accuracy: 0.5546
Epoch 171/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6835 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2421 - val_sparse_categorical_accuracy: 0.5677
Epoch 172/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6761 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2556 - val_sparse_categorical_accuracy: 0.5415
Epoch 173/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6798 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2728 - val_sparse_categorical_accuracy: 0.5459
Epoch 174/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6780 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2571 - val_sparse_categorical_accuracy: 0.5939
Epoch 175/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6796 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2367 - val_sparse_categorical_accuracy: 0.5895
Epoch 176/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6801 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2553 - val_sparse_categorical_accuracy: 0.5415
Epoch 177/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6762 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2694 - val_sparse_categorical_accuracy: 0.5546
Epoch 178/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6791 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2543 - val_sparse_categorical_accuracy: 0.5633
Epoch 179/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6778 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2253 - val_sparse_categorical_accuracy: 0.5721
Epoch 180/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6791 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2442 - val_sparse_categorical_accuracy: 0.5502
Epoch 181/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6790 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2805 - val_sparse_categorical_accuracy: 0.5328
Epoch 182/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6814 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2697 - val_sparse_categorical_accuracy: 0.5721
Epoch 183/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6767 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2514 - val_sparse_categorical_accuracy: 0.5633
Epoch 184/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6770 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2567 - val_sparse_categorical_accuracy: 0.5284
Epoch 185/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6770 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2689 - val_sparse_categorical_accuracy: 0.5459
Epoch 186/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6782 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2646 - val_sparse_categorical_accuracy: 0.5633
Epoch 187/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6779 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2468 - val_sparse_categorical_accuracy: 0.5852
Epoch 188/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6770 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2268 - val_sparse_categorical_accuracy: 0.5764
Epoch 189/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6815 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2350 - val_sparse_categorical_accuracy: 0.5721
Epoch 190/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6803 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2706 - val_sparse_categorical_accuracy: 0.5677
Epoch 191/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6788 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2883 - val_sparse_categorical_accuracy: 0.5633
Epoch 192/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6797 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2689 - val_sparse_categorical_accuracy: 0.5502
Epoch 193/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6745 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2520 - val_sparse_categorical_accuracy: 0.5764
Epoch 194/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6776 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2522 - val_sparse_categorical_accuracy: 0.5721
Epoch 195/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6773 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2662 - val_sparse_categorical_accuracy: 0.5590
Epoch 196/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6752 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2739 - val_sparse_categorical_accuracy: 0.5590
Epoch 197/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6761 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2648 - val_sparse_categorical_accuracy: 0.5371
Epoch 198/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6733 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2853 - val_sparse_categorical_accuracy: 0.5371
Epoch 199/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6798 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2653 - val_sparse_categorical_accuracy: 0.5677
Epoch 200/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6745 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2583 - val_sparse_categorical_accuracy: 0.5764
Epoch 201/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6787 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2629 - val_sparse_categorical_accuracy: 0.5590
Epoch 202/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6718 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2636 - val_sparse_categorical_accuracy: 0.5546
Epoch 203/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6766 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2443 - val_sparse_categorical_accuracy: 0.5677
Epoch 204/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6750 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2441 - val_sparse_categorical_accuracy: 0.5633
Epoch 205/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6767 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2599 - val_sparse_categorical_accuracy: 0.5459
Epoch 206/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6748 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2712 - val_sparse_categorical_accuracy: 0.5546
Epoch 207/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6735 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2638 - val_sparse_categorical_accuracy: 0.5502
Epoch 208/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6734 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.2491 - val_sparse_categorical_accuracy: 0.5546
Epoch 209/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6744 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2430 - val_sparse_categorical_accuracy: 0.5677
Epoch 210/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6802 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2507 - val_sparse_categorical_accuracy: 0.5852
Epoch 211/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6751 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2711 - val_sparse_categorical_accuracy: 0.5284
Epoch 212/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6750 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2647 - val_sparse_categorical_accuracy: 0.5721
Epoch 213/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6713 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2554 - val_sparse_categorical_accuracy: 0.5895
Epoch 214/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6775 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.2644 - val_sparse_categorical_accuracy: 0.5546
Epoch 215/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6748 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2779 - val_sparse_categorical_accuracy: 0.5240
Epoch 216/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6829 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2486 - val_sparse_categorical_accuracy: 0.5721
Epoch 217/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6750 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2533 - val_sparse_categorical_accuracy: 0.6114
Epoch 218/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6797 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2552 - val_sparse_categorical_accuracy: 0.5677
Epoch 219/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6724 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2715 - val_sparse_categorical_accuracy: 0.5328
Epoch 220/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6807 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2590 - val_sparse_categorical_accuracy: 0.5633
Epoch 221/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6756 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2472 - val_sparse_categorical_accuracy: 0.5764
Epoch 222/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6772 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2405 - val_sparse_categorical_accuracy: 0.5808
Epoch 223/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6749 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.2487 - val_sparse_categorical_accuracy: 0.5590
Epoch 224/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6747 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2578 - val_sparse_categorical_accuracy: 0.5677
Epoch 225/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6756 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2646 - val_sparse_categorical_accuracy: 0.5633
Epoch 226/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6743 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2591 - val_sparse_categorical_accuracy: 0.5546
Epoch 227/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6741 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2805 - val_sparse_categorical_accuracy: 0.5502
Epoch 228/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6738 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.3089 - val_sparse_categorical_accuracy: 0.5459
Epoch 229/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6781 - sparse_categorical_accuracy: 0.7007 - val_loss: 1.2698 - val_sparse_categorical_accuracy: 0.5633
Epoch 230/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6789 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2263 - val_sparse_categorical_accuracy: 0.5721
Epoch 231/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6781 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2417 - val_sparse_categorical_accuracy: 0.5371
Epoch 232/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6746 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2819 - val_sparse_categorical_accuracy: 0.5546
Epoch 233/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6808 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2641 - val_sparse_categorical_accuracy: 0.5764
Epoch 234/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6740 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2503 - val_sparse_categorical_accuracy: 0.5721
Epoch 235/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6817 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2864 - val_sparse_categorical_accuracy: 0.5240
Epoch 236/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6802 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2978 - val_sparse_categorical_accuracy: 0.5590
Epoch 237/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6758 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2653 - val_sparse_categorical_accuracy: 0.5808
Epoch 238/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6730 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2564 - val_sparse_categorical_accuracy: 0.5808
Epoch 239/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6734 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2577 - val_sparse_categorical_accuracy: 0.5852
Epoch 240/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6748 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2589 - val_sparse_categorical_accuracy: 0.5808
Epoch 241/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6743 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2514 - val_sparse_categorical_accuracy: 0.5590
Epoch 242/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6719 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2718 - val_sparse_categorical_accuracy: 0.5546
Epoch 243/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6756 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2848 - val_sparse_categorical_accuracy: 0.5502
Epoch 244/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6752 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2821 - val_sparse_categorical_accuracy: 0.5415
Epoch 245/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6752 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2530 - val_sparse_categorical_accuracy: 0.5677
Epoch 246/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6732 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2591 - val_sparse_categorical_accuracy: 0.5590
Epoch 247/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6717 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2680 - val_sparse_categorical_accuracy: 0.5546
Epoch 248/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6731 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2686 - val_sparse_categorical_accuracy: 0.5546
Epoch 249/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6732 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2647 - val_sparse_categorical_accuracy: 0.5546
Epoch 250/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6729 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2586 - val_sparse_categorical_accuracy: 0.5677
Epoch 251/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6722 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2652 - val_sparse_categorical_accuracy: 0.5633
Epoch 252/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6714 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2882 - val_sparse_categorical_accuracy: 0.5721
Epoch 253/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6720 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.2810 - val_sparse_categorical_accuracy: 0.5721
Epoch 254/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6722 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2657 - val_sparse_categorical_accuracy: 0.5764
Epoch 255/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6756 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2578 - val_sparse_categorical_accuracy: 0.5633
Epoch 256/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6732 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2749 - val_sparse_categorical_accuracy: 0.5502
Epoch 257/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6755 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2696 - val_sparse_categorical_accuracy: 0.5633
Epoch 258/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6719 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2663 - val_sparse_categorical_accuracy: 0.5852
Epoch 259/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6763 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2572 - val_sparse_categorical_accuracy: 0.5852
Epoch 260/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6719 - sparse_categorical_accuracy: 0.6978 - val_loss: 1.2534 - val_sparse_categorical_accuracy: 0.5721
Epoch 261/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6726 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2503 - val_sparse_categorical_accuracy: 0.5677
Epoch 262/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6720 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2642 - val_sparse_categorical_accuracy: 0.5721
Epoch 263/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6712 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2845 - val_sparse_categorical_accuracy: 0.5415
Epoch 264/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6735 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2947 - val_sparse_categorical_accuracy: 0.5546
Epoch 265/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6735 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2795 - val_sparse_categorical_accuracy: 0.5546
Epoch 266/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6712 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2690 - val_sparse_categorical_accuracy: 0.5677
Epoch 267/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6726 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2504 - val_sparse_categorical_accuracy: 0.5546
Epoch 268/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6729 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.2500 - val_sparse_categorical_accuracy: 0.5983
Epoch 269/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6713 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2727 - val_sparse_categorical_accuracy: 0.5852
Epoch 270/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6747 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2836 - val_sparse_categorical_accuracy: 0.5764
Epoch 271/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6698 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2960 - val_sparse_categorical_accuracy: 0.5415
Epoch 272/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6758 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2804 - val_sparse_categorical_accuracy: 0.5633
Epoch 273/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6707 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2607 - val_sparse_categorical_accuracy: 0.5633
Epoch 274/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6851 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2807 - val_sparse_categorical_accuracy: 0.5633
Epoch 275/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6734 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3255 - val_sparse_categorical_accuracy: 0.5328
Epoch 276/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6855 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2710 - val_sparse_categorical_accuracy: 0.5721
Epoch 277/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6745 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2293 - val_sparse_categorical_accuracy: 0.6070
Epoch 278/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6855 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2629 - val_sparse_categorical_accuracy: 0.5764
Epoch 279/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6755 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.3071 - val_sparse_categorical_accuracy: 0.5721
Epoch 280/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6784 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2796 - val_sparse_categorical_accuracy: 0.5590
Epoch 281/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6765 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2578 - val_sparse_categorical_accuracy: 0.5721
Epoch 282/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6732 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2832 - val_sparse_categorical_accuracy: 0.5459
Epoch 283/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6757 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2891 - val_sparse_categorical_accuracy: 0.5546
Epoch 284/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6743 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2759 - val_sparse_categorical_accuracy: 0.5677
Epoch 285/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6767 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2783 - val_sparse_categorical_accuracy: 0.5590
Epoch 286/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6751 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2898 - val_sparse_categorical_accuracy: 0.5590
Epoch 287/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6734 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2800 - val_sparse_categorical_accuracy: 0.5546
Epoch 288/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6725 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2649 - val_sparse_categorical_accuracy: 0.5546
Epoch 289/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6711 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.2824 - val_sparse_categorical_accuracy: 0.5677
Epoch 290/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6710 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2878 - val_sparse_categorical_accuracy: 0.5852
Epoch 291/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6709 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2566 - val_sparse_categorical_accuracy: 0.5721
Epoch 292/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6734 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2530 - val_sparse_categorical_accuracy: 0.5415
Epoch 293/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6746 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2906 - val_sparse_categorical_accuracy: 0.5415
Epoch 294/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6754 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3077 - val_sparse_categorical_accuracy: 0.5677
Epoch 295/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6732 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.2936 - val_sparse_categorical_accuracy: 0.5633
Epoch 296/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6783 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2805 - val_sparse_categorical_accuracy: 0.5502
Epoch 297/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6734 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.2881 - val_sparse_categorical_accuracy: 0.5677
Epoch 298/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6767 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2790 - val_sparse_categorical_accuracy: 0.5764
Epoch 299/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6714 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2932 - val_sparse_categorical_accuracy: 0.5415
Epoch 300/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6751 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2812 - val_sparse_categorical_accuracy: 0.5633
Epoch 301/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6669 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2655 - val_sparse_categorical_accuracy: 0.5895
Epoch 302/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6711 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2631 - val_sparse_categorical_accuracy: 0.5895
Epoch 303/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6687 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2893 - val_sparse_categorical_accuracy: 0.5502
Epoch 304/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6708 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2771 - val_sparse_categorical_accuracy: 0.5590
Epoch 305/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6696 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2671 - val_sparse_categorical_accuracy: 0.5633
Epoch 306/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6718 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2849 - val_sparse_categorical_accuracy: 0.5633
Epoch 307/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6682 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2864 - val_sparse_categorical_accuracy: 0.5590
Epoch 308/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6677 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2647 - val_sparse_categorical_accuracy: 0.5721
Epoch 309/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6677 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2718 - val_sparse_categorical_accuracy: 0.5852
Epoch 310/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6669 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2893 - val_sparse_categorical_accuracy: 0.5590
Epoch 311/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6693 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.2972 - val_sparse_categorical_accuracy: 0.5590
Epoch 312/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6675 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3084 - val_sparse_categorical_accuracy: 0.5633
Epoch 313/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6703 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2974 - val_sparse_categorical_accuracy: 0.5633
Epoch 314/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6690 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2880 - val_sparse_categorical_accuracy: 0.5328
Epoch 315/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6697 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2947 - val_sparse_categorical_accuracy: 0.5590
Epoch 316/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6679 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2779 - val_sparse_categorical_accuracy: 0.5721
Epoch 317/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6716 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2703 - val_sparse_categorical_accuracy: 0.5677
Epoch 318/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6701 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2874 - val_sparse_categorical_accuracy: 0.5546
Epoch 319/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6707 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2838 - val_sparse_categorical_accuracy: 0.5502
Epoch 320/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6698 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.2663 - val_sparse_categorical_accuracy: 0.5677
Epoch 321/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6726 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2824 - val_sparse_categorical_accuracy: 0.5590
Epoch 322/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6687 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.3114 - val_sparse_categorical_accuracy: 0.5546
Epoch 323/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6748 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2957 - val_sparse_categorical_accuracy: 0.5633
Epoch 324/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6712 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2683 - val_sparse_categorical_accuracy: 0.5764
Epoch 325/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6731 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2730 - val_sparse_categorical_accuracy: 0.5764
Epoch 326/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6757 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2938 - val_sparse_categorical_accuracy: 0.5721
Epoch 327/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6727 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2855 - val_sparse_categorical_accuracy: 0.5371
Epoch 328/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6724 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2640 - val_sparse_categorical_accuracy: 0.5546
Epoch 329/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6700 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2809 - val_sparse_categorical_accuracy: 0.5764
Epoch 330/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6714 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3065 - val_sparse_categorical_accuracy: 0.5590
Epoch 331/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6716 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2872 - val_sparse_categorical_accuracy: 0.5415
Epoch 332/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6698 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2787 - val_sparse_categorical_accuracy: 0.5546
Epoch 333/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6706 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3047 - val_sparse_categorical_accuracy: 0.5590
Epoch 334/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6690 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3042 - val_sparse_categorical_accuracy: 0.5721
Epoch 335/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6709 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2733 - val_sparse_categorical_accuracy: 0.5895
Epoch 336/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6697 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2771 - val_sparse_categorical_accuracy: 0.5764
Epoch 337/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6671 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2903 - val_sparse_categorical_accuracy: 0.5371
Epoch 338/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6678 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.2934 - val_sparse_categorical_accuracy: 0.5284
Epoch 339/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6732 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2773 - val_sparse_categorical_accuracy: 0.5502
Epoch 340/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6692 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2729 - val_sparse_categorical_accuracy: 0.5808
Epoch 341/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6744 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2981 - val_sparse_categorical_accuracy: 0.5764
Epoch 342/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6688 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3015 - val_sparse_categorical_accuracy: 0.5502
Epoch 343/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6703 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2745 - val_sparse_categorical_accuracy: 0.5677
Epoch 344/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6686 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2738 - val_sparse_categorical_accuracy: 0.5808
Epoch 345/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6684 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2895 - val_sparse_categorical_accuracy: 0.5546
Epoch 346/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6654 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3049 - val_sparse_categorical_accuracy: 0.5502
Epoch 347/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6717 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3003 - val_sparse_categorical_accuracy: 0.5590
Epoch 348/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6694 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2874 - val_sparse_categorical_accuracy: 0.5721
Epoch 349/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6675 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2775 - val_sparse_categorical_accuracy: 0.5459
Epoch 350/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6716 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3078 - val_sparse_categorical_accuracy: 0.5240
Epoch 351/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6697 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3249 - val_sparse_categorical_accuracy: 0.5459
Epoch 352/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6706 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2868 - val_sparse_categorical_accuracy: 0.5808
Epoch 353/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6730 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.2626 - val_sparse_categorical_accuracy: 0.5677
Epoch 354/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6774 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2878 - val_sparse_categorical_accuracy: 0.5153
Epoch 355/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6717 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2982 - val_sparse_categorical_accuracy: 0.5633
Epoch 356/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6715 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2912 - val_sparse_categorical_accuracy: 0.5852
Epoch 357/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6671 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2845 - val_sparse_categorical_accuracy: 0.5328
Epoch 358/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6726 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.2887 - val_sparse_categorical_accuracy: 0.5284
Epoch 359/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6714 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.2939 - val_sparse_categorical_accuracy: 0.5677
Epoch 360/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6735 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2960 - val_sparse_categorical_accuracy: 0.5721
Epoch 361/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6700 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2970 - val_sparse_categorical_accuracy: 0.5502
Epoch 362/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6737 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3019 - val_sparse_categorical_accuracy: 0.5240
Epoch 363/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6739 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2888 - val_sparse_categorical_accuracy: 0.5502
Epoch 364/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6680 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2802 - val_sparse_categorical_accuracy: 0.5677
Epoch 365/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6666 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2770 - val_sparse_categorical_accuracy: 0.5590
Epoch 366/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6652 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2763 - val_sparse_categorical_accuracy: 0.5721
Epoch 367/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6667 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2813 - val_sparse_categorical_accuracy: 0.6026
Epoch 368/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6683 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2751 - val_sparse_categorical_accuracy: 0.5808
Epoch 369/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6697 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2837 - val_sparse_categorical_accuracy: 0.5633
Epoch 370/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6664 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3184 - val_sparse_categorical_accuracy: 0.5590
Epoch 371/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6710 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3022 - val_sparse_categorical_accuracy: 0.5590
Epoch 372/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6698 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2964 - val_sparse_categorical_accuracy: 0.5677
Epoch 373/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6713 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.3234 - val_sparse_categorical_accuracy: 0.5415
Epoch 374/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6680 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3271 - val_sparse_categorical_accuracy: 0.5546
Epoch 375/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6687 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2879 - val_sparse_categorical_accuracy: 0.5764
Epoch 376/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6712 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2658 - val_sparse_categorical_accuracy: 0.5808
Epoch 377/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6693 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2919 - val_sparse_categorical_accuracy: 0.5633
Epoch 378/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6688 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3140 - val_sparse_categorical_accuracy: 0.5590
Epoch 379/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6687 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3031 - val_sparse_categorical_accuracy: 0.5633
Epoch 380/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6707 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2917 - val_sparse_categorical_accuracy: 0.5721
Epoch 381/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6673 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3206 - val_sparse_categorical_accuracy: 0.5546
Epoch 382/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6698 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3239 - val_sparse_categorical_accuracy: 0.5590
Epoch 383/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6711 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2812 - val_sparse_categorical_accuracy: 0.5721
Epoch 384/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6696 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.2891 - val_sparse_categorical_accuracy: 0.5633
Epoch 385/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6704 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3257 - val_sparse_categorical_accuracy: 0.5590
Epoch 386/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6657 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3155 - val_sparse_categorical_accuracy: 0.5721
Epoch 387/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6686 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2767 - val_sparse_categorical_accuracy: 0.5895
Epoch 388/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6705 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.2643 - val_sparse_categorical_accuracy: 0.5895
Epoch 389/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6740 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.3124 - val_sparse_categorical_accuracy: 0.5546
Epoch 390/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6684 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3395 - val_sparse_categorical_accuracy: 0.5502
Epoch 391/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6680 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.2973 - val_sparse_categorical_accuracy: 0.5502
Epoch 392/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6714 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2881 - val_sparse_categorical_accuracy: 0.5721
Epoch 393/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6732 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3203 - val_sparse_categorical_accuracy: 0.5502
Epoch 394/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6705 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3446 - val_sparse_categorical_accuracy: 0.5371
Epoch 395/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6685 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3025 - val_sparse_categorical_accuracy: 0.5677
Epoch 396/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6696 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2680 - val_sparse_categorical_accuracy: 0.5808
Epoch 397/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6726 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2849 - val_sparse_categorical_accuracy: 0.5590
Epoch 398/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6711 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3239 - val_sparse_categorical_accuracy: 0.5328
Epoch 399/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6795 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.3041 - val_sparse_categorical_accuracy: 0.5502
Epoch 400/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6666 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3093 - val_sparse_categorical_accuracy: 0.5852
Epoch 401/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6687 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3136 - val_sparse_categorical_accuracy: 0.5721
Epoch 402/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6672 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3170 - val_sparse_categorical_accuracy: 0.5546
Epoch 403/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6655 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.2849 - val_sparse_categorical_accuracy: 0.5677
Epoch 404/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6669 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.2762 - val_sparse_categorical_accuracy: 0.5721
Epoch 405/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6685 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3137 - val_sparse_categorical_accuracy: 0.5415
Epoch 406/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6660 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3429 - val_sparse_categorical_accuracy: 0.5459
Epoch 407/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6681 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3113 - val_sparse_categorical_accuracy: 0.5633
Epoch 408/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6675 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2914 - val_sparse_categorical_accuracy: 0.5721
Epoch 409/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6676 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3170 - val_sparse_categorical_accuracy: 0.5328
Epoch 410/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6648 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3295 - val_sparse_categorical_accuracy: 0.5459
Epoch 411/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6642 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3049 - val_sparse_categorical_accuracy: 0.5895
Epoch 412/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6665 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2927 - val_sparse_categorical_accuracy: 0.5895
Epoch 413/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6666 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2917 - val_sparse_categorical_accuracy: 0.5633
Epoch 414/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6676 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.2809 - val_sparse_categorical_accuracy: 0.5808
Epoch 415/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6666 - sparse_categorical_accuracy: 0.7080 - val_loss: 1.2943 - val_sparse_categorical_accuracy: 0.5852
Epoch 416/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6645 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3180 - val_sparse_categorical_accuracy: 0.5371
Epoch 417/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6643 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3170 - val_sparse_categorical_accuracy: 0.5459
Epoch 418/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6669 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3071 - val_sparse_categorical_accuracy: 0.5459
Epoch 419/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6633 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3176 - val_sparse_categorical_accuracy: 0.5633
Epoch 420/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6647 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3131 - val_sparse_categorical_accuracy: 0.5590
Epoch 421/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6640 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3016 - val_sparse_categorical_accuracy: 0.5677
Epoch 422/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6673 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2967 - val_sparse_categorical_accuracy: 0.5852
Epoch 423/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6642 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3081 - val_sparse_categorical_accuracy: 0.5546
Epoch 424/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6641 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3070 - val_sparse_categorical_accuracy: 0.5371
Epoch 425/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6649 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.2922 - val_sparse_categorical_accuracy: 0.5852
Epoch 426/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6652 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3088 - val_sparse_categorical_accuracy: 0.5808
Epoch 427/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6643 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3255 - val_sparse_categorical_accuracy: 0.5328
Epoch 428/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6644 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3163 - val_sparse_categorical_accuracy: 0.5371
Epoch 429/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6676 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3094 - val_sparse_categorical_accuracy: 0.5633
Epoch 430/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6641 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3124 - val_sparse_categorical_accuracy: 0.5721
Epoch 431/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6637 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3178 - val_sparse_categorical_accuracy: 0.5677
Epoch 432/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6637 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3415 - val_sparse_categorical_accuracy: 0.5459
Epoch 433/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6658 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3262 - val_sparse_categorical_accuracy: 0.5633
Epoch 434/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6636 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3018 - val_sparse_categorical_accuracy: 0.5895
Epoch 435/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6642 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3153 - val_sparse_categorical_accuracy: 0.5633
Epoch 436/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6654 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3276 - val_sparse_categorical_accuracy: 0.5459
Epoch 437/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6654 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3259 - val_sparse_categorical_accuracy: 0.5633
Epoch 438/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6649 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3442 - val_sparse_categorical_accuracy: 0.5459
Epoch 439/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6651 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3385 - val_sparse_categorical_accuracy: 0.5459
Epoch 440/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6637 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3002 - val_sparse_categorical_accuracy: 0.5852
Epoch 441/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6655 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3001 - val_sparse_categorical_accuracy: 0.5808
Epoch 442/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6638 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3220 - val_sparse_categorical_accuracy: 0.5633
Epoch 443/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6673 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3054 - val_sparse_categorical_accuracy: 0.5633
Epoch 444/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6652 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2884 - val_sparse_categorical_accuracy: 0.5677
Epoch 445/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6675 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3077 - val_sparse_categorical_accuracy: 0.5546
Epoch 446/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6658 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3418 - val_sparse_categorical_accuracy: 0.5328
Epoch 447/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6690 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3197 - val_sparse_categorical_accuracy: 0.5721
Epoch 448/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6650 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2969 - val_sparse_categorical_accuracy: 0.5764
Epoch 449/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6667 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2997 - val_sparse_categorical_accuracy: 0.5633
Epoch 450/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6649 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2973 - val_sparse_categorical_accuracy: 0.5415
Epoch 451/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6660 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.2853 - val_sparse_categorical_accuracy: 0.5633
Epoch 452/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6707 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.2896 - val_sparse_categorical_accuracy: 0.5721
Epoch 453/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6637 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3317 - val_sparse_categorical_accuracy: 0.5328
Epoch 454/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6692 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3335 - val_sparse_categorical_accuracy: 0.5590
Epoch 455/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6668 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.2895 - val_sparse_categorical_accuracy: 0.5895
Epoch 456/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6693 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2949 - val_sparse_categorical_accuracy: 0.5852
Epoch 457/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6673 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3300 - val_sparse_categorical_accuracy: 0.5677
Epoch 458/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6673 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3300 - val_sparse_categorical_accuracy: 0.5546
Epoch 459/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6637 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3233 - val_sparse_categorical_accuracy: 0.5808
Epoch 460/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6671 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.3369 - val_sparse_categorical_accuracy: 0.5721
Epoch 461/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6677 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3540 - val_sparse_categorical_accuracy: 0.5721
Epoch 462/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6737 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.3112 - val_sparse_categorical_accuracy: 0.5546
Epoch 463/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6663 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.2825 - val_sparse_categorical_accuracy: 0.5546
Epoch 464/2000
2/2 [==============================] - 0s 24ms/step - loss: 0.6837 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.2898 - val_sparse_categorical_accuracy: 0.5633
Epoch 465/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6699 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.3432 - val_sparse_categorical_accuracy: 0.5764
Epoch 466/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6744 - sparse_categorical_accuracy: 0.7051 - val_loss: 1.3379 - val_sparse_categorical_accuracy: 0.5633
Epoch 467/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6654 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2836 - val_sparse_categorical_accuracy: 0.5590
Epoch 468/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6777 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.2779 - val_sparse_categorical_accuracy: 0.5721
Epoch 469/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6724 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3259 - val_sparse_categorical_accuracy: 0.5895
Epoch 470/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6717 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.3601 - val_sparse_categorical_accuracy: 0.5721
Epoch 471/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6760 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3083 - val_sparse_categorical_accuracy: 0.5502
Epoch 472/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6755 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.2760 - val_sparse_categorical_accuracy: 0.5633
Epoch 473/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6722 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3086 - val_sparse_categorical_accuracy: 0.5852
Epoch 474/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6676 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3468 - val_sparse_categorical_accuracy: 0.5633
Epoch 475/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6668 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3249 - val_sparse_categorical_accuracy: 0.5240
Epoch 476/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6706 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.2890 - val_sparse_categorical_accuracy: 0.5546
Epoch 477/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6688 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3083 - val_sparse_categorical_accuracy: 0.5502
Epoch 478/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6630 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3514 - val_sparse_categorical_accuracy: 0.5415
Epoch 479/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6657 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3469 - val_sparse_categorical_accuracy: 0.5371
Epoch 480/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6682 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.2994 - val_sparse_categorical_accuracy: 0.5852
Epoch 481/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6641 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3085 - val_sparse_categorical_accuracy: 0.6026
Epoch 482/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6665 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3386 - val_sparse_categorical_accuracy: 0.5590
Epoch 483/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6631 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3409 - val_sparse_categorical_accuracy: 0.5415
Epoch 484/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6656 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3312 - val_sparse_categorical_accuracy: 0.5546
Epoch 485/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6656 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3185 - val_sparse_categorical_accuracy: 0.5677
Epoch 486/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6686 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3040 - val_sparse_categorical_accuracy: 0.5764
Epoch 487/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6634 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3335 - val_sparse_categorical_accuracy: 0.5677
Epoch 488/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6678 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3419 - val_sparse_categorical_accuracy: 0.5677
Epoch 489/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6627 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3260 - val_sparse_categorical_accuracy: 0.5764
Epoch 490/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6635 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3487 - val_sparse_categorical_accuracy: 0.5633
Epoch 491/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6686 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3579 - val_sparse_categorical_accuracy: 0.5459
Epoch 492/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6667 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3419 - val_sparse_categorical_accuracy: 0.5459
Epoch 493/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6611 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3346 - val_sparse_categorical_accuracy: 0.5808
Epoch 494/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6634 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3291 - val_sparse_categorical_accuracy: 0.5590
Epoch 495/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6617 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3213 - val_sparse_categorical_accuracy: 0.5459
Epoch 496/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6648 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3320 - val_sparse_categorical_accuracy: 0.5590
Epoch 497/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6656 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3461 - val_sparse_categorical_accuracy: 0.5677
Epoch 498/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6626 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3432 - val_sparse_categorical_accuracy: 0.5502
Epoch 499/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6627 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3279 - val_sparse_categorical_accuracy: 0.5546
Epoch 500/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6627 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3222 - val_sparse_categorical_accuracy: 0.5764
Epoch 501/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6607 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3282 - val_sparse_categorical_accuracy: 0.5677
Epoch 502/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6594 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3555 - val_sparse_categorical_accuracy: 0.5328
Epoch 503/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6648 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3609 - val_sparse_categorical_accuracy: 0.5328
Epoch 504/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6624 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3466 - val_sparse_categorical_accuracy: 0.5808
Epoch 505/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6639 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3314 - val_sparse_categorical_accuracy: 0.5852
Epoch 506/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6633 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3165 - val_sparse_categorical_accuracy: 0.5677
Epoch 507/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6604 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3185 - val_sparse_categorical_accuracy: 0.5415
Epoch 508/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6611 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3306 - val_sparse_categorical_accuracy: 0.5590
Epoch 509/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6590 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3341 - val_sparse_categorical_accuracy: 0.5546
Epoch 510/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6589 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3397 - val_sparse_categorical_accuracy: 0.5590
Epoch 511/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6607 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3603 - val_sparse_categorical_accuracy: 0.5502
Epoch 512/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6605 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3554 - val_sparse_categorical_accuracy: 0.5415
Epoch 513/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6602 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3269 - val_sparse_categorical_accuracy: 0.5633
Epoch 514/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6589 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3096 - val_sparse_categorical_accuracy: 0.5939
Epoch 515/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6681 - sparse_categorical_accuracy: 0.7066 - val_loss: 1.3159 - val_sparse_categorical_accuracy: 0.5721
Epoch 516/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6615 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3419 - val_sparse_categorical_accuracy: 0.5546
Epoch 517/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6600 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3451 - val_sparse_categorical_accuracy: 0.5633
Epoch 518/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6615 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3348 - val_sparse_categorical_accuracy: 0.5852
Epoch 519/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6597 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3537 - val_sparse_categorical_accuracy: 0.5502
Epoch 520/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6610 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3546 - val_sparse_categorical_accuracy: 0.5284
Epoch 521/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6598 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3374 - val_sparse_categorical_accuracy: 0.5808
Epoch 522/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6623 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3274 - val_sparse_categorical_accuracy: 0.5721
Epoch 523/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6655 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3294 - val_sparse_categorical_accuracy: 0.5459
Epoch 524/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6618 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3315 - val_sparse_categorical_accuracy: 0.5328
Epoch 525/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6641 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3284 - val_sparse_categorical_accuracy: 0.5721
Epoch 526/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6607 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3459 - val_sparse_categorical_accuracy: 0.5590
Epoch 527/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6715 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.3233 - val_sparse_categorical_accuracy: 0.5546
Epoch 528/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6638 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3261 - val_sparse_categorical_accuracy: 0.5240
Epoch 529/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6663 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3289 - val_sparse_categorical_accuracy: 0.5677
Epoch 530/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6598 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3193 - val_sparse_categorical_accuracy: 0.5590
Epoch 531/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6635 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3182 - val_sparse_categorical_accuracy: 0.5546
Epoch 532/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6619 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3458 - val_sparse_categorical_accuracy: 0.5240
Epoch 533/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6643 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3360 - val_sparse_categorical_accuracy: 0.5677
Epoch 534/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6609 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3302 - val_sparse_categorical_accuracy: 0.5721
Epoch 535/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6643 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3436 - val_sparse_categorical_accuracy: 0.5371
Epoch 536/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6660 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3357 - val_sparse_categorical_accuracy: 0.5415
Epoch 537/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6634 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3104 - val_sparse_categorical_accuracy: 0.6070
Epoch 538/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6681 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2978 - val_sparse_categorical_accuracy: 0.5895
Epoch 539/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6641 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3558 - val_sparse_categorical_accuracy: 0.5371
Epoch 540/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6675 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3807 - val_sparse_categorical_accuracy: 0.5633
Epoch 541/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6657 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3507 - val_sparse_categorical_accuracy: 0.5677
Epoch 542/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6649 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3355 - val_sparse_categorical_accuracy: 0.5852
Epoch 543/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6605 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3653 - val_sparse_categorical_accuracy: 0.5284
Epoch 544/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6646 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3596 - val_sparse_categorical_accuracy: 0.5371
Epoch 545/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6617 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3213 - val_sparse_categorical_accuracy: 0.5808
Epoch 546/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6605 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3023 - val_sparse_categorical_accuracy: 0.5764
Epoch 547/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6612 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3308 - val_sparse_categorical_accuracy: 0.5502
Epoch 548/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6639 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3622 - val_sparse_categorical_accuracy: 0.5633
Epoch 549/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6671 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3428 - val_sparse_categorical_accuracy: 0.5546
Epoch 550/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6596 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3392 - val_sparse_categorical_accuracy: 0.5633
Epoch 551/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6646 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3440 - val_sparse_categorical_accuracy: 0.5502
Epoch 552/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6571 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3622 - val_sparse_categorical_accuracy: 0.5459
Epoch 553/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6649 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3245 - val_sparse_categorical_accuracy: 0.5502
Epoch 554/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6607 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3141 - val_sparse_categorical_accuracy: 0.5808
Epoch 555/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6673 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3584 - val_sparse_categorical_accuracy: 0.5808
Epoch 556/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6643 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3695 - val_sparse_categorical_accuracy: 0.5371
Epoch 557/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6598 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3293 - val_sparse_categorical_accuracy: 0.5590
Epoch 558/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6604 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3391 - val_sparse_categorical_accuracy: 0.5721
Epoch 559/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6615 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3650 - val_sparse_categorical_accuracy: 0.5415
Epoch 560/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6577 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3652 - val_sparse_categorical_accuracy: 0.5284
Epoch 561/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6665 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3626 - val_sparse_categorical_accuracy: 0.5546
Epoch 562/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6620 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3587 - val_sparse_categorical_accuracy: 0.5721
Epoch 563/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6599 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3347 - val_sparse_categorical_accuracy: 0.5677
Epoch 564/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6631 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3251 - val_sparse_categorical_accuracy: 0.5546
Epoch 565/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6629 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.3266 - val_sparse_categorical_accuracy: 0.5721
Epoch 566/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6592 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3544 - val_sparse_categorical_accuracy: 0.5764
Epoch 567/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6598 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3591 - val_sparse_categorical_accuracy: 0.5677
Epoch 568/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6567 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3336 - val_sparse_categorical_accuracy: 0.5808
Epoch 569/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6594 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3379 - val_sparse_categorical_accuracy: 0.5808
Epoch 570/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6605 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3587 - val_sparse_categorical_accuracy: 0.5590
Epoch 571/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6569 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3638 - val_sparse_categorical_accuracy: 0.5284
Epoch 572/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6607 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3606 - val_sparse_categorical_accuracy: 0.5546
Epoch 573/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6614 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3724 - val_sparse_categorical_accuracy: 0.5808
Epoch 574/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6636 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3600 - val_sparse_categorical_accuracy: 0.5459
Epoch 575/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6609 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3643 - val_sparse_categorical_accuracy: 0.5415
Epoch 576/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6690 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3582 - val_sparse_categorical_accuracy: 0.5459
Epoch 577/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6599 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3386 - val_sparse_categorical_accuracy: 0.5764
Epoch 578/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6651 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3388 - val_sparse_categorical_accuracy: 0.5895
Epoch 579/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6630 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3550 - val_sparse_categorical_accuracy: 0.5328
Epoch 580/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6606 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3717 - val_sparse_categorical_accuracy: 0.5459
Epoch 581/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6572 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3686 - val_sparse_categorical_accuracy: 0.5677
Epoch 582/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6622 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3458 - val_sparse_categorical_accuracy: 0.5764
Epoch 583/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6618 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3316 - val_sparse_categorical_accuracy: 0.5633
Epoch 584/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6598 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3448 - val_sparse_categorical_accuracy: 0.5371
Epoch 585/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6602 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3336 - val_sparse_categorical_accuracy: 0.5764
Epoch 586/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6578 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3206 - val_sparse_categorical_accuracy: 0.5852
Epoch 587/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6628 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3604 - val_sparse_categorical_accuracy: 0.5459
Epoch 588/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6574 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4160 - val_sparse_categorical_accuracy: 0.5371
Epoch 589/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6647 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3795 - val_sparse_categorical_accuracy: 0.5328
Epoch 590/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6572 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3456 - val_sparse_categorical_accuracy: 0.5677
Epoch 591/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6599 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3511 - val_sparse_categorical_accuracy: 0.5546
Epoch 592/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6617 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3471 - val_sparse_categorical_accuracy: 0.5459
Epoch 593/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6591 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3459 - val_sparse_categorical_accuracy: 0.5502
Epoch 594/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6573 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3842 - val_sparse_categorical_accuracy: 0.5546
Epoch 595/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6628 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3744 - val_sparse_categorical_accuracy: 0.5459
Epoch 596/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6579 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3183 - val_sparse_categorical_accuracy: 0.5546
Epoch 597/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6688 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3010 - val_sparse_categorical_accuracy: 0.5852
Epoch 598/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6628 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3373 - val_sparse_categorical_accuracy: 0.5764
Epoch 599/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6644 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3541 - val_sparse_categorical_accuracy: 0.5546
Epoch 600/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6626 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3261 - val_sparse_categorical_accuracy: 0.5633
Epoch 601/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6622 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3360 - val_sparse_categorical_accuracy: 0.5633
Epoch 602/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6611 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3822 - val_sparse_categorical_accuracy: 0.5328
Epoch 603/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6600 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3867 - val_sparse_categorical_accuracy: 0.5371
Epoch 604/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6610 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3480 - val_sparse_categorical_accuracy: 0.5633
Epoch 605/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6564 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3575 - val_sparse_categorical_accuracy: 0.5721
Epoch 606/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6565 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3746 - val_sparse_categorical_accuracy: 0.5590
Epoch 607/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6566 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3615 - val_sparse_categorical_accuracy: 0.5459
Epoch 608/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6552 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3641 - val_sparse_categorical_accuracy: 0.5502
Epoch 609/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6573 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3734 - val_sparse_categorical_accuracy: 0.5371
Epoch 610/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6568 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3712 - val_sparse_categorical_accuracy: 0.5371
Epoch 611/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6561 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3596 - val_sparse_categorical_accuracy: 0.5546
Epoch 612/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6595 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3651 - val_sparse_categorical_accuracy: 0.5459
Epoch 613/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6572 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3729 - val_sparse_categorical_accuracy: 0.5459
Epoch 614/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6560 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3645 - val_sparse_categorical_accuracy: 0.5852
Epoch 615/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6546 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3354 - val_sparse_categorical_accuracy: 0.5808
Epoch 616/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6597 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3406 - val_sparse_categorical_accuracy: 0.5502
Epoch 617/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6584 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3704 - val_sparse_categorical_accuracy: 0.5371
Epoch 618/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6615 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3596 - val_sparse_categorical_accuracy: 0.5590
Epoch 619/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6559 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3701 - val_sparse_categorical_accuracy: 0.5546
Epoch 620/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6564 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3911 - val_sparse_categorical_accuracy: 0.5284
Epoch 621/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6591 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3773 - val_sparse_categorical_accuracy: 0.5284
Epoch 622/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6556 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3433 - val_sparse_categorical_accuracy: 0.5764
Epoch 623/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6614 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3477 - val_sparse_categorical_accuracy: 0.5677
Epoch 624/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6593 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3958 - val_sparse_categorical_accuracy: 0.5197
Epoch 625/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6617 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3662 - val_sparse_categorical_accuracy: 0.5764
Epoch 626/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6572 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3324 - val_sparse_categorical_accuracy: 0.5852
Epoch 627/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6591 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3452 - val_sparse_categorical_accuracy: 0.5415
Epoch 628/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6591 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3891 - val_sparse_categorical_accuracy: 0.5197
Epoch 629/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6621 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3828 - val_sparse_categorical_accuracy: 0.5633
Epoch 630/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6548 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3689 - val_sparse_categorical_accuracy: 0.5895
Epoch 631/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6584 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3764 - val_sparse_categorical_accuracy: 0.5590
Epoch 632/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6583 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3857 - val_sparse_categorical_accuracy: 0.5284
Epoch 633/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6569 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3594 - val_sparse_categorical_accuracy: 0.5677
Epoch 634/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6544 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3389 - val_sparse_categorical_accuracy: 0.5895
Epoch 635/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6613 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3666 - val_sparse_categorical_accuracy: 0.5895
Epoch 636/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6566 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4078 - val_sparse_categorical_accuracy: 0.5459
Epoch 637/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6647 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3579 - val_sparse_categorical_accuracy: 0.5284
Epoch 638/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6618 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3187 - val_sparse_categorical_accuracy: 0.5852
Epoch 639/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6711 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3635 - val_sparse_categorical_accuracy: 0.5633
Epoch 640/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6622 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.4223 - val_sparse_categorical_accuracy: 0.5284
Epoch 641/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6638 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3682 - val_sparse_categorical_accuracy: 0.5546
Epoch 642/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6594 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3208 - val_sparse_categorical_accuracy: 0.5590
Epoch 643/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6648 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3162 - val_sparse_categorical_accuracy: 0.5764
Epoch 644/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6593 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3492 - val_sparse_categorical_accuracy: 0.5502
Epoch 645/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6616 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3392 - val_sparse_categorical_accuracy: 0.5677
Epoch 646/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6554 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3494 - val_sparse_categorical_accuracy: 0.5764
Epoch 647/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6588 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3821 - val_sparse_categorical_accuracy: 0.5633
Epoch 648/2000
2/2 [==============================] - 0s 23ms/step - loss: 0.6556 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3788 - val_sparse_categorical_accuracy: 0.5808
Epoch 649/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6582 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3598 - val_sparse_categorical_accuracy: 0.5590
Epoch 650/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6560 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3668 - val_sparse_categorical_accuracy: 0.5415
Epoch 651/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6558 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3753 - val_sparse_categorical_accuracy: 0.5546
Epoch 652/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6531 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3710 - val_sparse_categorical_accuracy: 0.5808
Epoch 653/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6549 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3679 - val_sparse_categorical_accuracy: 0.5415
Epoch 654/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6544 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3888 - val_sparse_categorical_accuracy: 0.5240
Epoch 655/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6596 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3693 - val_sparse_categorical_accuracy: 0.5590
Epoch 656/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6573 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3802 - val_sparse_categorical_accuracy: 0.5808
Epoch 657/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6598 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3893 - val_sparse_categorical_accuracy: 0.5328
Epoch 658/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6566 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3635 - val_sparse_categorical_accuracy: 0.5415
Epoch 659/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6559 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3547 - val_sparse_categorical_accuracy: 0.5590
Epoch 660/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6566 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3685 - val_sparse_categorical_accuracy: 0.5590
Epoch 661/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6542 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3661 - val_sparse_categorical_accuracy: 0.5764
Epoch 662/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6566 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3431 - val_sparse_categorical_accuracy: 0.5633
Epoch 663/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6550 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3530 - val_sparse_categorical_accuracy: 0.5633
Epoch 664/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6537 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3939 - val_sparse_categorical_accuracy: 0.5502
Epoch 665/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6552 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4179 - val_sparse_categorical_accuracy: 0.5459
Epoch 666/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6565 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3858 - val_sparse_categorical_accuracy: 0.5459
Epoch 667/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6551 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.3730 - val_sparse_categorical_accuracy: 0.5546
Epoch 668/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6535 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.3766 - val_sparse_categorical_accuracy: 0.5677
Epoch 669/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6533 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3698 - val_sparse_categorical_accuracy: 0.5502
Epoch 670/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6574 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3721 - val_sparse_categorical_accuracy: 0.5371
Epoch 671/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6543 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3780 - val_sparse_categorical_accuracy: 0.5764
Epoch 672/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6521 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3971 - val_sparse_categorical_accuracy: 0.5284
Epoch 673/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6601 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3697 - val_sparse_categorical_accuracy: 0.5546
Epoch 674/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6624 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3405 - val_sparse_categorical_accuracy: 0.5895
Epoch 675/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6594 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3845 - val_sparse_categorical_accuracy: 0.5546
Epoch 676/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6582 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3755 - val_sparse_categorical_accuracy: 0.5284
Epoch 677/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6598 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3494 - val_sparse_categorical_accuracy: 0.5677
Epoch 678/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6635 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3649 - val_sparse_categorical_accuracy: 0.5502
Epoch 679/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6554 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4022 - val_sparse_categorical_accuracy: 0.5284
Epoch 680/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6587 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3633 - val_sparse_categorical_accuracy: 0.5590
Epoch 681/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6572 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3377 - val_sparse_categorical_accuracy: 0.5852
Epoch 682/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6621 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3808 - val_sparse_categorical_accuracy: 0.5677
Epoch 683/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6620 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4096 - val_sparse_categorical_accuracy: 0.5284
Epoch 684/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6574 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3654 - val_sparse_categorical_accuracy: 0.5633
Epoch 685/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6586 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3758 - val_sparse_categorical_accuracy: 0.5633
Epoch 686/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6547 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4108 - val_sparse_categorical_accuracy: 0.5328
Epoch 687/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6586 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4024 - val_sparse_categorical_accuracy: 0.5371
Epoch 688/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6529 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3955 - val_sparse_categorical_accuracy: 0.5328
Epoch 689/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6591 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3804 - val_sparse_categorical_accuracy: 0.5459
Epoch 690/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6533 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3879 - val_sparse_categorical_accuracy: 0.5764
Epoch 691/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6549 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3727 - val_sparse_categorical_accuracy: 0.5546
Epoch 692/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6528 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.3648 - val_sparse_categorical_accuracy: 0.5197
Epoch 693/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6663 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3832 - val_sparse_categorical_accuracy: 0.5502
Epoch 694/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6547 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4100 - val_sparse_categorical_accuracy: 0.5590
Epoch 695/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6599 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3768 - val_sparse_categorical_accuracy: 0.5459
Epoch 696/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6558 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3620 - val_sparse_categorical_accuracy: 0.5502
Epoch 697/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6593 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3818 - val_sparse_categorical_accuracy: 0.5546
Epoch 698/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6539 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3961 - val_sparse_categorical_accuracy: 0.5590
Epoch 699/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6586 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3604 - val_sparse_categorical_accuracy: 0.5546
Epoch 700/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6545 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3659 - val_sparse_categorical_accuracy: 0.5502
Epoch 701/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6532 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4008 - val_sparse_categorical_accuracy: 0.5590
Epoch 702/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6549 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3929 - val_sparse_categorical_accuracy: 0.5371
Epoch 703/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6530 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3615 - val_sparse_categorical_accuracy: 0.5633
Epoch 704/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6550 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3856 - val_sparse_categorical_accuracy: 0.5546
Epoch 705/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6522 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4153 - val_sparse_categorical_accuracy: 0.5415
Epoch 706/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6573 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3665 - val_sparse_categorical_accuracy: 0.5415
Epoch 707/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6544 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.3252 - val_sparse_categorical_accuracy: 0.5677
Epoch 708/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6635 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3019 - val_sparse_categorical_accuracy: 0.5764
Epoch 709/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6665 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.2767 - val_sparse_categorical_accuracy: 0.5852
Epoch 710/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6682 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.2728 - val_sparse_categorical_accuracy: 0.5852
Epoch 711/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6655 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.2834 - val_sparse_categorical_accuracy: 0.5808
Epoch 712/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6611 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3421 - val_sparse_categorical_accuracy: 0.5502
Epoch 713/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6565 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4066 - val_sparse_categorical_accuracy: 0.5109
Epoch 714/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6647 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.4048 - val_sparse_categorical_accuracy: 0.5459
Epoch 715/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6638 - sparse_categorical_accuracy: 0.7036 - val_loss: 1.3840 - val_sparse_categorical_accuracy: 0.5764
Epoch 716/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6616 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3537 - val_sparse_categorical_accuracy: 0.5808
Epoch 717/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6582 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3568 - val_sparse_categorical_accuracy: 0.5415
Epoch 718/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6652 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3230 - val_sparse_categorical_accuracy: 0.5808
Epoch 719/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6565 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3138 - val_sparse_categorical_accuracy: 0.6026
Epoch 720/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6652 - sparse_categorical_accuracy: 0.7109 - val_loss: 1.3365 - val_sparse_categorical_accuracy: 0.5764
Epoch 721/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6538 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3916 - val_sparse_categorical_accuracy: 0.5197
Epoch 722/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6600 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4109 - val_sparse_categorical_accuracy: 0.5328
Epoch 723/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6572 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3679 - val_sparse_categorical_accuracy: 0.5721
Epoch 724/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6614 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3513 - val_sparse_categorical_accuracy: 0.5721
Epoch 725/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6558 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3715 - val_sparse_categorical_accuracy: 0.5371
Epoch 726/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6580 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3788 - val_sparse_categorical_accuracy: 0.5546
Epoch 727/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6576 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3843 - val_sparse_categorical_accuracy: 0.5677
Epoch 728/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6537 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4018 - val_sparse_categorical_accuracy: 0.5721
Epoch 729/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6557 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4060 - val_sparse_categorical_accuracy: 0.5415
Epoch 730/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6539 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3854 - val_sparse_categorical_accuracy: 0.5459
Epoch 731/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6574 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3586 - val_sparse_categorical_accuracy: 0.5502
Epoch 732/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6543 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.3458 - val_sparse_categorical_accuracy: 0.5633
Epoch 733/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6523 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3617 - val_sparse_categorical_accuracy: 0.5764
Epoch 734/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6551 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3710 - val_sparse_categorical_accuracy: 0.5633
Epoch 735/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6510 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3848 - val_sparse_categorical_accuracy: 0.5546
Epoch 736/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6529 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4117 - val_sparse_categorical_accuracy: 0.5502
Epoch 737/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6542 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4039 - val_sparse_categorical_accuracy: 0.5721
Epoch 738/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6552 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3871 - val_sparse_categorical_accuracy: 0.5677
Epoch 739/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6502 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3847 - val_sparse_categorical_accuracy: 0.5502
Epoch 740/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6514 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3676 - val_sparse_categorical_accuracy: 0.5764
Epoch 741/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6535 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3423 - val_sparse_categorical_accuracy: 0.5895
Epoch 742/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6578 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3625 - val_sparse_categorical_accuracy: 0.5459
Epoch 743/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6513 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4239 - val_sparse_categorical_accuracy: 0.5153
Epoch 744/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6631 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4148 - val_sparse_categorical_accuracy: 0.5371
Epoch 745/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6525 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3803 - val_sparse_categorical_accuracy: 0.5590
Epoch 746/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6620 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3851 - val_sparse_categorical_accuracy: 0.5502
Epoch 747/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6520 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4224 - val_sparse_categorical_accuracy: 0.5459
Epoch 748/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6562 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4028 - val_sparse_categorical_accuracy: 0.5459
Epoch 749/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6503 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3825 - val_sparse_categorical_accuracy: 0.5721
Epoch 750/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6583 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3959 - val_sparse_categorical_accuracy: 0.5764
Epoch 751/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6502 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.4114 - val_sparse_categorical_accuracy: 0.5371
Epoch 752/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6534 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3906 - val_sparse_categorical_accuracy: 0.5328
Epoch 753/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6549 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3904 - val_sparse_categorical_accuracy: 0.5459
Epoch 754/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6524 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4193 - val_sparse_categorical_accuracy: 0.5502
Epoch 755/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6523 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4327 - val_sparse_categorical_accuracy: 0.5371
Epoch 756/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6524 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3768 - val_sparse_categorical_accuracy: 0.5502
Epoch 757/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6549 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3375 - val_sparse_categorical_accuracy: 0.5808
Epoch 758/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6632 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3820 - val_sparse_categorical_accuracy: 0.5371
Epoch 759/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6543 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4249 - val_sparse_categorical_accuracy: 0.5240
Epoch 760/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6552 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3945 - val_sparse_categorical_accuracy: 0.5546
Epoch 761/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6540 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3670 - val_sparse_categorical_accuracy: 0.5764
Epoch 762/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6523 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3896 - val_sparse_categorical_accuracy: 0.5371
Epoch 763/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6540 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4007 - val_sparse_categorical_accuracy: 0.5590
Epoch 764/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6524 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3906 - val_sparse_categorical_accuracy: 0.5721
Epoch 765/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6537 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4059 - val_sparse_categorical_accuracy: 0.5590
Epoch 766/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6544 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4132 - val_sparse_categorical_accuracy: 0.5546
Epoch 767/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6512 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4032 - val_sparse_categorical_accuracy: 0.5371
Epoch 768/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6518 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3912 - val_sparse_categorical_accuracy: 0.5590
Epoch 769/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6486 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3959 - val_sparse_categorical_accuracy: 0.5983
Epoch 770/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6584 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3883 - val_sparse_categorical_accuracy: 0.5546
Epoch 771/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6527 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3874 - val_sparse_categorical_accuracy: 0.5153
Epoch 772/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6543 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3789 - val_sparse_categorical_accuracy: 0.5459
Epoch 773/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6503 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3960 - val_sparse_categorical_accuracy: 0.5677
Epoch 774/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6587 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4055 - val_sparse_categorical_accuracy: 0.5371
Epoch 775/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6499 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4139 - val_sparse_categorical_accuracy: 0.5284
Epoch 776/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6548 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4043 - val_sparse_categorical_accuracy: 0.5677
Epoch 777/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6538 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3943 - val_sparse_categorical_accuracy: 0.5721
Epoch 778/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6579 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3768 - val_sparse_categorical_accuracy: 0.5677
Epoch 779/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6555 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3858 - val_sparse_categorical_accuracy: 0.5284
Epoch 780/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6673 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3828 - val_sparse_categorical_accuracy: 0.5590
Epoch 781/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6556 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3738 - val_sparse_categorical_accuracy: 0.5633
Epoch 782/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6631 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3753 - val_sparse_categorical_accuracy: 0.5546
Epoch 783/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6557 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4022 - val_sparse_categorical_accuracy: 0.5284
Epoch 784/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6548 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4011 - val_sparse_categorical_accuracy: 0.5459
Epoch 785/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6523 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3755 - val_sparse_categorical_accuracy: 0.5721
Epoch 786/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6517 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3646 - val_sparse_categorical_accuracy: 0.5677
Epoch 787/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6517 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3854 - val_sparse_categorical_accuracy: 0.5764
Epoch 788/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6528 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3989 - val_sparse_categorical_accuracy: 0.5633
Epoch 789/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6506 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.3876 - val_sparse_categorical_accuracy: 0.5502
Epoch 790/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6545 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3967 - val_sparse_categorical_accuracy: 0.5415
Epoch 791/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6478 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4325 - val_sparse_categorical_accuracy: 0.5590
Epoch 792/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6608 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3732 - val_sparse_categorical_accuracy: 0.5852
Epoch 793/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6577 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3437 - val_sparse_categorical_accuracy: 0.5764
Epoch 794/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6645 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3941 - val_sparse_categorical_accuracy: 0.5546
Epoch 795/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6540 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4291 - val_sparse_categorical_accuracy: 0.5546
Epoch 796/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6603 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.3943 - val_sparse_categorical_accuracy: 0.5459
Epoch 797/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6495 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3870 - val_sparse_categorical_accuracy: 0.5284
Epoch 798/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6564 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3938 - val_sparse_categorical_accuracy: 0.5502
Epoch 799/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6523 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4035 - val_sparse_categorical_accuracy: 0.5721
Epoch 800/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6548 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3919 - val_sparse_categorical_accuracy: 0.5633
Epoch 801/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6495 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4109 - val_sparse_categorical_accuracy: 0.5284
Epoch 802/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6609 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4321 - val_sparse_categorical_accuracy: 0.5328
Epoch 803/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6529 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4425 - val_sparse_categorical_accuracy: 0.5764
Epoch 804/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6591 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3977 - val_sparse_categorical_accuracy: 0.5808
Epoch 805/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6512 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3730 - val_sparse_categorical_accuracy: 0.5371
Epoch 806/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6551 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.3810 - val_sparse_categorical_accuracy: 0.5633
Epoch 807/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6517 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3936 - val_sparse_categorical_accuracy: 0.5764
Epoch 808/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6533 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3954 - val_sparse_categorical_accuracy: 0.5764
Epoch 809/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6545 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.3974 - val_sparse_categorical_accuracy: 0.5415
Epoch 810/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6507 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3988 - val_sparse_categorical_accuracy: 0.5371
Epoch 811/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6529 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3812 - val_sparse_categorical_accuracy: 0.5459
Epoch 812/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6485 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3905 - val_sparse_categorical_accuracy: 0.5546
Epoch 813/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6489 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3983 - val_sparse_categorical_accuracy: 0.5721
Epoch 814/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6511 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4004 - val_sparse_categorical_accuracy: 0.5677
Epoch 815/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6489 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4206 - val_sparse_categorical_accuracy: 0.5415
Epoch 816/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6512 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4149 - val_sparse_categorical_accuracy: 0.5328
Epoch 817/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6521 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4027 - val_sparse_categorical_accuracy: 0.5415
Epoch 818/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6508 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3824 - val_sparse_categorical_accuracy: 0.5415
Epoch 819/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6521 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4058 - val_sparse_categorical_accuracy: 0.5415
Epoch 820/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6504 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4136 - val_sparse_categorical_accuracy: 0.5459
Epoch 821/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6475 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.3834 - val_sparse_categorical_accuracy: 0.5546
Epoch 822/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6530 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3841 - val_sparse_categorical_accuracy: 0.5677
Epoch 823/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6550 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.4294 - val_sparse_categorical_accuracy: 0.5502
Epoch 824/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6518 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4404 - val_sparse_categorical_accuracy: 0.5459
Epoch 825/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6561 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4072 - val_sparse_categorical_accuracy: 0.5677
Epoch 826/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6498 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3957 - val_sparse_categorical_accuracy: 0.5764
Epoch 827/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6547 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4117 - val_sparse_categorical_accuracy: 0.5502
Epoch 828/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6496 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4405 - val_sparse_categorical_accuracy: 0.5240
Epoch 829/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6563 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.4018 - val_sparse_categorical_accuracy: 0.5502
Epoch 830/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6488 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3851 - val_sparse_categorical_accuracy: 0.5677
Epoch 831/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6552 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4065 - val_sparse_categorical_accuracy: 0.5677
Epoch 832/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6462 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4308 - val_sparse_categorical_accuracy: 0.5328
Epoch 833/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6566 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4128 - val_sparse_categorical_accuracy: 0.5371
Epoch 834/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6544 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3885 - val_sparse_categorical_accuracy: 0.5677
Epoch 835/2000
2/2 [==============================] - 0s 21ms/step - loss: 0.6512 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3921 - val_sparse_categorical_accuracy: 0.5677
Epoch 836/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6502 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4009 - val_sparse_categorical_accuracy: 0.5371
Epoch 837/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6487 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3972 - val_sparse_categorical_accuracy: 0.5371
Epoch 838/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6471 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.3844 - val_sparse_categorical_accuracy: 0.5808
Epoch 839/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6496 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4043 - val_sparse_categorical_accuracy: 0.5852
Epoch 840/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6494 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.4122 - val_sparse_categorical_accuracy: 0.5371
Epoch 841/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6477 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4026 - val_sparse_categorical_accuracy: 0.5328
Epoch 842/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6467 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4071 - val_sparse_categorical_accuracy: 0.5677
Epoch 843/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6500 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4167 - val_sparse_categorical_accuracy: 0.5590
Epoch 844/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6451 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4294 - val_sparse_categorical_accuracy: 0.5197
Epoch 845/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6547 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4008 - val_sparse_categorical_accuracy: 0.5633
Epoch 846/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6498 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3900 - val_sparse_categorical_accuracy: 0.5590
Epoch 847/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6514 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4235 - val_sparse_categorical_accuracy: 0.5721
Epoch 848/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6492 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4423 - val_sparse_categorical_accuracy: 0.5240
Epoch 849/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6489 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4130 - val_sparse_categorical_accuracy: 0.5721
Epoch 850/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6464 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4116 - val_sparse_categorical_accuracy: 0.5808
Epoch 851/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6563 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4084 - val_sparse_categorical_accuracy: 0.5633
Epoch 852/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6454 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4251 - val_sparse_categorical_accuracy: 0.5066
Epoch 853/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6579 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4046 - val_sparse_categorical_accuracy: 0.5371
Epoch 854/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6496 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4153 - val_sparse_categorical_accuracy: 0.5546
Epoch 855/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6561 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4178 - val_sparse_categorical_accuracy: 0.5677
Epoch 856/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6482 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4170 - val_sparse_categorical_accuracy: 0.5240
Epoch 857/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6538 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4208 - val_sparse_categorical_accuracy: 0.5284
Epoch 858/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6485 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4030 - val_sparse_categorical_accuracy: 0.5808
Epoch 859/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6526 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3752 - val_sparse_categorical_accuracy: 0.5721
Epoch 860/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6525 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3922 - val_sparse_categorical_accuracy: 0.5590
Epoch 861/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6458 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4439 - val_sparse_categorical_accuracy: 0.5240
Epoch 862/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6542 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4137 - val_sparse_categorical_accuracy: 0.5459
Epoch 863/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6477 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.3950 - val_sparse_categorical_accuracy: 0.5721
Epoch 864/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6560 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4332 - val_sparse_categorical_accuracy: 0.5633
Epoch 865/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6479 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4399 - val_sparse_categorical_accuracy: 0.5153
Epoch 866/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6514 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3728 - val_sparse_categorical_accuracy: 0.5546
Epoch 867/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6490 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3498 - val_sparse_categorical_accuracy: 0.5895
Epoch 868/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6579 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3895 - val_sparse_categorical_accuracy: 0.5677
Epoch 869/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6467 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4421 - val_sparse_categorical_accuracy: 0.5153
Epoch 870/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6557 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.4319 - val_sparse_categorical_accuracy: 0.5371
Epoch 871/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6493 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4062 - val_sparse_categorical_accuracy: 0.5764
Epoch 872/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6514 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4080 - val_sparse_categorical_accuracy: 0.5764
Epoch 873/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6479 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4030 - val_sparse_categorical_accuracy: 0.5415
Epoch 874/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6538 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.3829 - val_sparse_categorical_accuracy: 0.5808
Epoch 875/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6494 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4087 - val_sparse_categorical_accuracy: 0.5808
Epoch 876/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6559 - sparse_categorical_accuracy: 0.7139 - val_loss: 1.3892 - val_sparse_categorical_accuracy: 0.5590
Epoch 877/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6480 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.3618 - val_sparse_categorical_accuracy: 0.5677
Epoch 878/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6635 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3822 - val_sparse_categorical_accuracy: 0.5677
Epoch 879/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6599 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4379 - val_sparse_categorical_accuracy: 0.5328
Epoch 880/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6604 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4422 - val_sparse_categorical_accuracy: 0.5415
Epoch 881/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6591 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.4219 - val_sparse_categorical_accuracy: 0.5328
Epoch 882/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6528 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.4057 - val_sparse_categorical_accuracy: 0.5459
Epoch 883/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6506 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3997 - val_sparse_categorical_accuracy: 0.5590
Epoch 884/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6537 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3808 - val_sparse_categorical_accuracy: 0.5590
Epoch 885/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6511 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3917 - val_sparse_categorical_accuracy: 0.5546
Epoch 886/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6488 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4202 - val_sparse_categorical_accuracy: 0.5502
Epoch 887/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6490 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4525 - val_sparse_categorical_accuracy: 0.5502
Epoch 888/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6527 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4413 - val_sparse_categorical_accuracy: 0.5459
Epoch 889/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6498 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4143 - val_sparse_categorical_accuracy: 0.5590
Epoch 890/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6509 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4094 - val_sparse_categorical_accuracy: 0.5546
Epoch 891/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6525 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.3957 - val_sparse_categorical_accuracy: 0.5415
Epoch 892/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6489 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3910 - val_sparse_categorical_accuracy: 0.5459
Epoch 893/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6482 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4121 - val_sparse_categorical_accuracy: 0.5546
Epoch 894/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6484 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4200 - val_sparse_categorical_accuracy: 0.5328
Epoch 895/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6480 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.4152 - val_sparse_categorical_accuracy: 0.5328
Epoch 896/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6477 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3907 - val_sparse_categorical_accuracy: 0.5764
Epoch 897/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6486 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.3927 - val_sparse_categorical_accuracy: 0.5721
Epoch 898/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6468 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4134 - val_sparse_categorical_accuracy: 0.5546
Epoch 899/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6450 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4212 - val_sparse_categorical_accuracy: 0.5328
Epoch 900/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6477 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3792 - val_sparse_categorical_accuracy: 0.5633
Epoch 901/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6495 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3708 - val_sparse_categorical_accuracy: 0.5721
Epoch 902/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6497 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.4178 - val_sparse_categorical_accuracy: 0.5677
Epoch 903/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6478 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4338 - val_sparse_categorical_accuracy: 0.5240
Epoch 904/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6476 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.4056 - val_sparse_categorical_accuracy: 0.5546
Epoch 905/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6490 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.3926 - val_sparse_categorical_accuracy: 0.5677
Epoch 906/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6516 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4092 - val_sparse_categorical_accuracy: 0.5721
Epoch 907/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6500 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4200 - val_sparse_categorical_accuracy: 0.5590
Epoch 908/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6435 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4244 - val_sparse_categorical_accuracy: 0.5677
Epoch 909/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6473 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4221 - val_sparse_categorical_accuracy: 0.5590
Epoch 910/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6440 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4220 - val_sparse_categorical_accuracy: 0.5371
Epoch 911/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6458 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4123 - val_sparse_categorical_accuracy: 0.5459
Epoch 912/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6449 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4126 - val_sparse_categorical_accuracy: 0.5459
Epoch 913/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6444 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4372 - val_sparse_categorical_accuracy: 0.5721
Epoch 914/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6460 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4457 - val_sparse_categorical_accuracy: 0.5721
Epoch 915/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6521 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4075 - val_sparse_categorical_accuracy: 0.5502
Epoch 916/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6455 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4155 - val_sparse_categorical_accuracy: 0.5284
Epoch 917/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6507 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4329 - val_sparse_categorical_accuracy: 0.5284
Epoch 918/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6444 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4225 - val_sparse_categorical_accuracy: 0.5721
Epoch 919/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6458 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4145 - val_sparse_categorical_accuracy: 0.5633
Epoch 920/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6437 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4368 - val_sparse_categorical_accuracy: 0.5459
Epoch 921/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6444 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.4278 - val_sparse_categorical_accuracy: 0.5633
Epoch 922/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6415 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4102 - val_sparse_categorical_accuracy: 0.5721
Epoch 923/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6454 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4195 - val_sparse_categorical_accuracy: 0.5502
Epoch 924/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6462 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4475 - val_sparse_categorical_accuracy: 0.5371
Epoch 925/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6464 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4523 - val_sparse_categorical_accuracy: 0.5546
Epoch 926/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6438 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4165 - val_sparse_categorical_accuracy: 0.5590
Epoch 927/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6473 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4093 - val_sparse_categorical_accuracy: 0.5677
Epoch 928/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6439 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.4455 - val_sparse_categorical_accuracy: 0.5502
Epoch 929/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6493 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4238 - val_sparse_categorical_accuracy: 0.5546
Epoch 930/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4134 - val_sparse_categorical_accuracy: 0.5764
Epoch 931/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6481 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4479 - val_sparse_categorical_accuracy: 0.5721
Epoch 932/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6469 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4513 - val_sparse_categorical_accuracy: 0.5633
Epoch 933/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6464 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4114 - val_sparse_categorical_accuracy: 0.5371
Epoch 934/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6550 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.3949 - val_sparse_categorical_accuracy: 0.5502
Epoch 935/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6533 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4438 - val_sparse_categorical_accuracy: 0.5764
Epoch 936/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6524 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4337 - val_sparse_categorical_accuracy: 0.5633
Epoch 937/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6442 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3912 - val_sparse_categorical_accuracy: 0.5764
Epoch 938/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6571 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4182 - val_sparse_categorical_accuracy: 0.5197
Epoch 939/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6612 - sparse_categorical_accuracy: 0.7095 - val_loss: 1.4686 - val_sparse_categorical_accuracy: 0.5240
Epoch 940/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6490 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4341 - val_sparse_categorical_accuracy: 0.5590
Epoch 941/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6563 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4083 - val_sparse_categorical_accuracy: 0.5459
Epoch 942/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6519 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4432 - val_sparse_categorical_accuracy: 0.5197
Epoch 943/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6540 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4512 - val_sparse_categorical_accuracy: 0.5415
Epoch 944/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6484 - sparse_categorical_accuracy: 0.7153 - val_loss: 1.4175 - val_sparse_categorical_accuracy: 0.5590
Epoch 945/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6487 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4187 - val_sparse_categorical_accuracy: 0.5415
Epoch 946/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6476 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4373 - val_sparse_categorical_accuracy: 0.5240
Epoch 947/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6479 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4341 - val_sparse_categorical_accuracy: 0.5459
Epoch 948/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6468 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4165 - val_sparse_categorical_accuracy: 0.5764
Epoch 949/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6514 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4084 - val_sparse_categorical_accuracy: 0.5677
Epoch 950/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6482 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4516 - val_sparse_categorical_accuracy: 0.5415
Epoch 951/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6492 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4590 - val_sparse_categorical_accuracy: 0.5502
Epoch 952/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6487 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.3940 - val_sparse_categorical_accuracy: 0.5284
Epoch 953/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6501 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.3697 - val_sparse_categorical_accuracy: 0.5328
Epoch 954/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6503 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3811 - val_sparse_categorical_accuracy: 0.5721
Epoch 955/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6488 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.3896 - val_sparse_categorical_accuracy: 0.5939
Epoch 956/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6465 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.3865 - val_sparse_categorical_accuracy: 0.5764
Epoch 957/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6493 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4375 - val_sparse_categorical_accuracy: 0.5502
Epoch 958/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6470 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4708 - val_sparse_categorical_accuracy: 0.5633
Epoch 959/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6470 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4367 - val_sparse_categorical_accuracy: 0.5895
Epoch 960/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6469 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4146 - val_sparse_categorical_accuracy: 0.5677
Epoch 961/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6482 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4239 - val_sparse_categorical_accuracy: 0.5546
Epoch 962/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6449 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4379 - val_sparse_categorical_accuracy: 0.5284
Epoch 963/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6460 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4259 - val_sparse_categorical_accuracy: 0.5197
Epoch 964/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6457 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4185 - val_sparse_categorical_accuracy: 0.5677
Epoch 965/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6448 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4277 - val_sparse_categorical_accuracy: 0.5590
Epoch 966/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6422 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4520 - val_sparse_categorical_accuracy: 0.5502
Epoch 967/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6441 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4639 - val_sparse_categorical_accuracy: 0.5502
Epoch 968/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6450 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4371 - val_sparse_categorical_accuracy: 0.5721
Epoch 969/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6447 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4006 - val_sparse_categorical_accuracy: 0.5502
Epoch 970/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6437 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4126 - val_sparse_categorical_accuracy: 0.5371
Epoch 971/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6458 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4268 - val_sparse_categorical_accuracy: 0.5459
Epoch 972/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6446 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4082 - val_sparse_categorical_accuracy: 0.5590
Epoch 973/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6463 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4160 - val_sparse_categorical_accuracy: 0.5590
Epoch 974/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6492 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4718 - val_sparse_categorical_accuracy: 0.5328
Epoch 975/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6529 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.4549 - val_sparse_categorical_accuracy: 0.5502
Epoch 976/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6439 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3831 - val_sparse_categorical_accuracy: 0.5502
Epoch 977/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6552 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.3716 - val_sparse_categorical_accuracy: 0.5546
Epoch 978/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6524 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4100 - val_sparse_categorical_accuracy: 0.5590
Epoch 979/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6455 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4239 - val_sparse_categorical_accuracy: 0.5502
Epoch 980/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4274 - val_sparse_categorical_accuracy: 0.5415
Epoch 981/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6452 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4379 - val_sparse_categorical_accuracy: 0.5459
Epoch 982/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6434 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4726 - val_sparse_categorical_accuracy: 0.5240
Epoch 983/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6440 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4860 - val_sparse_categorical_accuracy: 0.5328
Epoch 984/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6441 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4340 - val_sparse_categorical_accuracy: 0.5371
Epoch 985/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6425 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.3988 - val_sparse_categorical_accuracy: 0.5415
Epoch 986/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6469 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4029 - val_sparse_categorical_accuracy: 0.5546
Epoch 987/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6436 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4117 - val_sparse_categorical_accuracy: 0.5852
Epoch 988/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6508 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4162 - val_sparse_categorical_accuracy: 0.5633
Epoch 989/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6479 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4505 - val_sparse_categorical_accuracy: 0.5284
Epoch 990/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6467 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4642 - val_sparse_categorical_accuracy: 0.5677
Epoch 991/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6438 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4614 - val_sparse_categorical_accuracy: 0.5721
Epoch 992/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6440 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4655 - val_sparse_categorical_accuracy: 0.5546
Epoch 993/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6488 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4601 - val_sparse_categorical_accuracy: 0.5328
Epoch 994/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4456 - val_sparse_categorical_accuracy: 0.5721
Epoch 995/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6458 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4410 - val_sparse_categorical_accuracy: 0.5677
Epoch 996/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6431 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4521 - val_sparse_categorical_accuracy: 0.5371
Epoch 997/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6436 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4471 - val_sparse_categorical_accuracy: 0.5371
Epoch 998/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6421 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4460 - val_sparse_categorical_accuracy: 0.5677
Epoch 999/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4575 - val_sparse_categorical_accuracy: 0.5764
Epoch 1000/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6438 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4626 - val_sparse_categorical_accuracy: 0.5284
Epoch 1001/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6415 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4596 - val_sparse_categorical_accuracy: 0.5502
Epoch 1002/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6415 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4580 - val_sparse_categorical_accuracy: 0.5721
Epoch 1003/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6412 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4710 - val_sparse_categorical_accuracy: 0.5371
Epoch 1004/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6410 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4716 - val_sparse_categorical_accuracy: 0.5415
Epoch 1005/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6404 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4451 - val_sparse_categorical_accuracy: 0.5502
Epoch 1006/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6412 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4131 - val_sparse_categorical_accuracy: 0.5590
Epoch 1007/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6439 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4022 - val_sparse_categorical_accuracy: 0.5633
Epoch 1008/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6454 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4430 - val_sparse_categorical_accuracy: 0.5721
Epoch 1009/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6411 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4615 - val_sparse_categorical_accuracy: 0.5328
Epoch 1010/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4563 - val_sparse_categorical_accuracy: 0.5371
Epoch 1011/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6455 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4753 - val_sparse_categorical_accuracy: 0.5459
Epoch 1012/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6408 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4925 - val_sparse_categorical_accuracy: 0.5328
Epoch 1013/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6431 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4640 - val_sparse_categorical_accuracy: 0.5371
Epoch 1014/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6404 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4242 - val_sparse_categorical_accuracy: 0.5502
Epoch 1015/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6412 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4096 - val_sparse_categorical_accuracy: 0.5590
Epoch 1016/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6435 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4285 - val_sparse_categorical_accuracy: 0.5590
Epoch 1017/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6421 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4309 - val_sparse_categorical_accuracy: 0.5677
Epoch 1018/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6408 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4385 - val_sparse_categorical_accuracy: 0.5677
Epoch 1019/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6428 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4688 - val_sparse_categorical_accuracy: 0.5459
Epoch 1020/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6484 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4790 - val_sparse_categorical_accuracy: 0.5197
Epoch 1021/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6505 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.4348 - val_sparse_categorical_accuracy: 0.5808
Epoch 1022/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4460 - val_sparse_categorical_accuracy: 0.5633
Epoch 1023/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6394 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4715 - val_sparse_categorical_accuracy: 0.5328
Epoch 1024/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6436 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4602 - val_sparse_categorical_accuracy: 0.5590
Epoch 1025/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6415 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4600 - val_sparse_categorical_accuracy: 0.5852
Epoch 1026/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6429 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4728 - val_sparse_categorical_accuracy: 0.5721
Epoch 1027/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4537 - val_sparse_categorical_accuracy: 0.5677
Epoch 1028/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6443 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.4554 - val_sparse_categorical_accuracy: 0.5240
Epoch 1029/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6424 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4695 - val_sparse_categorical_accuracy: 0.5590
Epoch 1030/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6480 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4373 - val_sparse_categorical_accuracy: 0.5590
Epoch 1031/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6435 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4293 - val_sparse_categorical_accuracy: 0.5240
Epoch 1032/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6479 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4688 - val_sparse_categorical_accuracy: 0.5240
Epoch 1033/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6406 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4720 - val_sparse_categorical_accuracy: 0.5808
Epoch 1034/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6458 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4475 - val_sparse_categorical_accuracy: 0.5808
Epoch 1035/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6414 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4684 - val_sparse_categorical_accuracy: 0.5240
Epoch 1036/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6501 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4654 - val_sparse_categorical_accuracy: 0.5197
Epoch 1037/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6390 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.4392 - val_sparse_categorical_accuracy: 0.5764
Epoch 1038/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6488 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4258 - val_sparse_categorical_accuracy: 0.5764
Epoch 1039/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6440 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4597 - val_sparse_categorical_accuracy: 0.5371
Epoch 1040/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6493 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4548 - val_sparse_categorical_accuracy: 0.5415
Epoch 1041/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6413 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4343 - val_sparse_categorical_accuracy: 0.5677
Epoch 1042/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6503 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4464 - val_sparse_categorical_accuracy: 0.5459
Epoch 1043/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6416 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4965 - val_sparse_categorical_accuracy: 0.5153
Epoch 1044/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6508 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4672 - val_sparse_categorical_accuracy: 0.5284
Epoch 1045/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6435 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4407 - val_sparse_categorical_accuracy: 0.5852
Epoch 1046/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6471 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4687 - val_sparse_categorical_accuracy: 0.5677
Epoch 1047/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6496 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4734 - val_sparse_categorical_accuracy: 0.5284
Epoch 1048/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6424 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4525 - val_sparse_categorical_accuracy: 0.5109
Epoch 1049/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6430 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4554 - val_sparse_categorical_accuracy: 0.5633
Epoch 1050/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6435 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4579 - val_sparse_categorical_accuracy: 0.5721
Epoch 1051/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6428 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4552 - val_sparse_categorical_accuracy: 0.5459
Epoch 1052/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6469 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4663 - val_sparse_categorical_accuracy: 0.5459
Epoch 1053/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6484 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4356 - val_sparse_categorical_accuracy: 0.5764
Epoch 1054/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6444 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4245 - val_sparse_categorical_accuracy: 0.5633
Epoch 1055/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6438 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4790 - val_sparse_categorical_accuracy: 0.5240
Epoch 1056/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6483 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4829 - val_sparse_categorical_accuracy: 0.5197
Epoch 1057/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6476 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4481 - val_sparse_categorical_accuracy: 0.5502
Epoch 1058/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6458 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4698 - val_sparse_categorical_accuracy: 0.5721
Epoch 1059/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6466 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.4850 - val_sparse_categorical_accuracy: 0.5240
Epoch 1060/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6440 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4736 - val_sparse_categorical_accuracy: 0.5240
Epoch 1061/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6444 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4517 - val_sparse_categorical_accuracy: 0.5808
Epoch 1062/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6402 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4336 - val_sparse_categorical_accuracy: 0.5721
Epoch 1063/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6412 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4430 - val_sparse_categorical_accuracy: 0.5284
Epoch 1064/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6425 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4780 - val_sparse_categorical_accuracy: 0.5240
Epoch 1065/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6408 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4771 - val_sparse_categorical_accuracy: 0.5677
Epoch 1066/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6412 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4584 - val_sparse_categorical_accuracy: 0.5502
Epoch 1067/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6396 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4590 - val_sparse_categorical_accuracy: 0.5284
Epoch 1068/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6394 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4621 - val_sparse_categorical_accuracy: 0.5502
Epoch 1069/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6438 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4515 - val_sparse_categorical_accuracy: 0.5633
Epoch 1070/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6406 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4452 - val_sparse_categorical_accuracy: 0.5502
Epoch 1071/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6441 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4703 - val_sparse_categorical_accuracy: 0.5459
Epoch 1072/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6451 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4682 - val_sparse_categorical_accuracy: 0.5808
Epoch 1073/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6396 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4487 - val_sparse_categorical_accuracy: 0.5764
Epoch 1074/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6370 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4710 - val_sparse_categorical_accuracy: 0.5197
Epoch 1075/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6414 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4792 - val_sparse_categorical_accuracy: 0.5197
Epoch 1076/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6385 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.4649 - val_sparse_categorical_accuracy: 0.5677
Epoch 1077/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6424 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4450 - val_sparse_categorical_accuracy: 0.5721
Epoch 1078/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6390 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4393 - val_sparse_categorical_accuracy: 0.5328
Epoch 1079/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6419 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.4519 - val_sparse_categorical_accuracy: 0.5371
Epoch 1080/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6427 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4434 - val_sparse_categorical_accuracy: 0.5852
Epoch 1081/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6403 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4423 - val_sparse_categorical_accuracy: 0.5764
Epoch 1082/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6488 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4689 - val_sparse_categorical_accuracy: 0.5371
Epoch 1083/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6414 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4980 - val_sparse_categorical_accuracy: 0.5371
Epoch 1084/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6463 - sparse_categorical_accuracy: 0.7168 - val_loss: 1.4474 - val_sparse_categorical_accuracy: 0.5546
Epoch 1085/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6426 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4324 - val_sparse_categorical_accuracy: 0.5546
Epoch 1086/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6490 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4691 - val_sparse_categorical_accuracy: 0.5328
Epoch 1087/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6421 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4857 - val_sparse_categorical_accuracy: 0.5371
Epoch 1088/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6403 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4659 - val_sparse_categorical_accuracy: 0.5633
Epoch 1089/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6386 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4912 - val_sparse_categorical_accuracy: 0.5459
Epoch 1090/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6414 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4844 - val_sparse_categorical_accuracy: 0.5328
Epoch 1091/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4523 - val_sparse_categorical_accuracy: 0.5633
Epoch 1092/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6379 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4547 - val_sparse_categorical_accuracy: 0.5721
Epoch 1093/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6450 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4415 - val_sparse_categorical_accuracy: 0.5546
Epoch 1094/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6377 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.4661 - val_sparse_categorical_accuracy: 0.5109
Epoch 1095/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6547 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4719 - val_sparse_categorical_accuracy: 0.5415
Epoch 1096/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6411 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4662 - val_sparse_categorical_accuracy: 0.5808
Epoch 1097/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6437 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4559 - val_sparse_categorical_accuracy: 0.5721
Epoch 1098/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6412 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4917 - val_sparse_categorical_accuracy: 0.5197
Epoch 1099/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6446 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5070 - val_sparse_categorical_accuracy: 0.5328
Epoch 1100/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6403 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.4638 - val_sparse_categorical_accuracy: 0.5852
Epoch 1101/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6402 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4430 - val_sparse_categorical_accuracy: 0.5808
Epoch 1102/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6383 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4727 - val_sparse_categorical_accuracy: 0.5546
Epoch 1103/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6434 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4412 - val_sparse_categorical_accuracy: 0.5328
Epoch 1104/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6424 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4344 - val_sparse_categorical_accuracy: 0.5415
Epoch 1105/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6463 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4960 - val_sparse_categorical_accuracy: 0.5240
Epoch 1106/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6397 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5337 - val_sparse_categorical_accuracy: 0.5197
Epoch 1107/2000
2/2 [==============================] - 0s 21ms/step - loss: 0.6425 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4677 - val_sparse_categorical_accuracy: 0.5371
Epoch 1108/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6384 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4085 - val_sparse_categorical_accuracy: 0.5764
Epoch 1109/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6438 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4242 - val_sparse_categorical_accuracy: 0.5808
Epoch 1110/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6401 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4556 - val_sparse_categorical_accuracy: 0.5677
Epoch 1111/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6389 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4621 - val_sparse_categorical_accuracy: 0.5677
Epoch 1112/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6378 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4646 - val_sparse_categorical_accuracy: 0.5590
Epoch 1113/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6387 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4958 - val_sparse_categorical_accuracy: 0.5415
Epoch 1114/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6419 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5257 - val_sparse_categorical_accuracy: 0.5328
Epoch 1115/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6410 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4618 - val_sparse_categorical_accuracy: 0.5502
Epoch 1116/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6431 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4456 - val_sparse_categorical_accuracy: 0.5633
Epoch 1117/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6485 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4798 - val_sparse_categorical_accuracy: 0.5546
Epoch 1118/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6411 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4847 - val_sparse_categorical_accuracy: 0.5415
Epoch 1119/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6401 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4666 - val_sparse_categorical_accuracy: 0.5328
Epoch 1120/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6383 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4805 - val_sparse_categorical_accuracy: 0.5764
Epoch 1121/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6360 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4953 - val_sparse_categorical_accuracy: 0.5721
Epoch 1122/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6385 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4887 - val_sparse_categorical_accuracy: 0.5808
Epoch 1123/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6366 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4745 - val_sparse_categorical_accuracy: 0.5764
Epoch 1124/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6360 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.4758 - val_sparse_categorical_accuracy: 0.5677
Epoch 1125/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6364 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4687 - val_sparse_categorical_accuracy: 0.5459
Epoch 1126/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6377 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4568 - val_sparse_categorical_accuracy: 0.5852
Epoch 1127/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6394 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4759 - val_sparse_categorical_accuracy: 0.5808
Epoch 1128/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6410 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4646 - val_sparse_categorical_accuracy: 0.5590
Epoch 1129/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6362 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4507 - val_sparse_categorical_accuracy: 0.5808
Epoch 1130/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6376 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4702 - val_sparse_categorical_accuracy: 0.5808
Epoch 1131/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6421 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4712 - val_sparse_categorical_accuracy: 0.5764
Epoch 1132/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6380 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4492 - val_sparse_categorical_accuracy: 0.5633
Epoch 1133/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6397 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.4364 - val_sparse_categorical_accuracy: 0.5590
Epoch 1134/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6402 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.4737 - val_sparse_categorical_accuracy: 0.5459
Epoch 1135/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6381 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5050 - val_sparse_categorical_accuracy: 0.5764
Epoch 1136/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6441 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4874 - val_sparse_categorical_accuracy: 0.5721
Epoch 1137/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6427 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4570 - val_sparse_categorical_accuracy: 0.5590
Epoch 1138/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6392 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4516 - val_sparse_categorical_accuracy: 0.5590
Epoch 1139/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6377 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.4797 - val_sparse_categorical_accuracy: 0.5721
Epoch 1140/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6399 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4791 - val_sparse_categorical_accuracy: 0.5721
Epoch 1141/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6397 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4776 - val_sparse_categorical_accuracy: 0.5721
Epoch 1142/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6379 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4780 - val_sparse_categorical_accuracy: 0.5546
Epoch 1143/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6360 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4688 - val_sparse_categorical_accuracy: 0.5459
Epoch 1144/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6381 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4390 - val_sparse_categorical_accuracy: 0.5459
Epoch 1145/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6377 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.4628 - val_sparse_categorical_accuracy: 0.5284
Epoch 1146/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6378 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5213 - val_sparse_categorical_accuracy: 0.5590
Epoch 1147/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6429 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5207 - val_sparse_categorical_accuracy: 0.5677
Epoch 1148/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6391 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4957 - val_sparse_categorical_accuracy: 0.5371
Epoch 1149/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6410 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4912 - val_sparse_categorical_accuracy: 0.5328
Epoch 1150/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6366 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4606 - val_sparse_categorical_accuracy: 0.5764
Epoch 1151/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6420 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4515 - val_sparse_categorical_accuracy: 0.5633
Epoch 1152/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6444 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4847 - val_sparse_categorical_accuracy: 0.5371
Epoch 1153/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6466 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4934 - val_sparse_categorical_accuracy: 0.5371
Epoch 1154/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6384 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4804 - val_sparse_categorical_accuracy: 0.5721
Epoch 1155/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6474 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.4527 - val_sparse_categorical_accuracy: 0.5590
Epoch 1156/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6402 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5037 - val_sparse_categorical_accuracy: 0.5197
Epoch 1157/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6506 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5188 - val_sparse_categorical_accuracy: 0.5153
Epoch 1158/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6388 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4991 - val_sparse_categorical_accuracy: 0.5721
Epoch 1159/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6405 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4554 - val_sparse_categorical_accuracy: 0.5808
Epoch 1160/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6406 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4824 - val_sparse_categorical_accuracy: 0.5415
Epoch 1161/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6354 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5237 - val_sparse_categorical_accuracy: 0.5197
Epoch 1162/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6385 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4824 - val_sparse_categorical_accuracy: 0.5546
Epoch 1163/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6359 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4663 - val_sparse_categorical_accuracy: 0.5764
Epoch 1164/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6397 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4916 - val_sparse_categorical_accuracy: 0.5415
Epoch 1165/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6340 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4867 - val_sparse_categorical_accuracy: 0.5371
Epoch 1166/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6348 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4824 - val_sparse_categorical_accuracy: 0.5502
Epoch 1167/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6336 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4865 - val_sparse_categorical_accuracy: 0.5764
Epoch 1168/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6343 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.4788 - val_sparse_categorical_accuracy: 0.5764
Epoch 1169/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6357 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4677 - val_sparse_categorical_accuracy: 0.5721
Epoch 1170/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6351 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5010 - val_sparse_categorical_accuracy: 0.5240
Epoch 1171/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6376 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5282 - val_sparse_categorical_accuracy: 0.5109
Epoch 1172/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6429 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4817 - val_sparse_categorical_accuracy: 0.5633
Epoch 1173/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6356 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4600 - val_sparse_categorical_accuracy: 0.5808
Epoch 1174/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6465 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.4803 - val_sparse_categorical_accuracy: 0.5721
Epoch 1175/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6409 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4881 - val_sparse_categorical_accuracy: 0.5197
Epoch 1176/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6434 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4548 - val_sparse_categorical_accuracy: 0.5371
Epoch 1177/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6424 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4736 - val_sparse_categorical_accuracy: 0.5633
Epoch 1178/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6401 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5155 - val_sparse_categorical_accuracy: 0.5677
Epoch 1179/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6376 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4876 - val_sparse_categorical_accuracy: 0.5371
Epoch 1180/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6378 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4568 - val_sparse_categorical_accuracy: 0.5546
Epoch 1181/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6392 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.4805 - val_sparse_categorical_accuracy: 0.5895
Epoch 1182/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6353 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4989 - val_sparse_categorical_accuracy: 0.5721
Epoch 1183/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6368 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4928 - val_sparse_categorical_accuracy: 0.5415
Epoch 1184/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6349 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5103 - val_sparse_categorical_accuracy: 0.5415
Epoch 1185/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6358 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5289 - val_sparse_categorical_accuracy: 0.5371
Epoch 1186/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6354 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5011 - val_sparse_categorical_accuracy: 0.5415
Epoch 1187/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6348 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4822 - val_sparse_categorical_accuracy: 0.5808
Epoch 1188/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6334 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5037 - val_sparse_categorical_accuracy: 0.5721
Epoch 1189/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6368 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4842 - val_sparse_categorical_accuracy: 0.5590
Epoch 1190/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6369 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4790 - val_sparse_categorical_accuracy: 0.5328
Epoch 1191/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6411 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5110 - val_sparse_categorical_accuracy: 0.5764
Epoch 1192/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6358 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5173 - val_sparse_categorical_accuracy: 0.5721
Epoch 1193/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6380 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4786 - val_sparse_categorical_accuracy: 0.5808
Epoch 1194/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6357 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4798 - val_sparse_categorical_accuracy: 0.5240
Epoch 1195/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6394 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4845 - val_sparse_categorical_accuracy: 0.5328
Epoch 1196/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6455 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4537 - val_sparse_categorical_accuracy: 0.5459
Epoch 1197/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6377 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4674 - val_sparse_categorical_accuracy: 0.5415
Epoch 1198/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6382 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5215 - val_sparse_categorical_accuracy: 0.5371
Epoch 1199/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6365 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5262 - val_sparse_categorical_accuracy: 0.5546
Epoch 1200/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6337 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4842 - val_sparse_categorical_accuracy: 0.5677
Epoch 1201/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6341 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4645 - val_sparse_categorical_accuracy: 0.5808
Epoch 1202/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6376 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4788 - val_sparse_categorical_accuracy: 0.5764
Epoch 1203/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6339 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5246 - val_sparse_categorical_accuracy: 0.5284
Epoch 1204/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6353 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5326 - val_sparse_categorical_accuracy: 0.5328
Epoch 1205/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6389 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5054 - val_sparse_categorical_accuracy: 0.5371
Epoch 1206/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6334 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4763 - val_sparse_categorical_accuracy: 0.5328
Epoch 1207/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6368 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4468 - val_sparse_categorical_accuracy: 0.5546
Epoch 1208/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6368 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4435 - val_sparse_categorical_accuracy: 0.5677
Epoch 1209/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6357 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4920 - val_sparse_categorical_accuracy: 0.5502
Epoch 1210/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6339 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5246 - val_sparse_categorical_accuracy: 0.5590
Epoch 1211/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6349 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4993 - val_sparse_categorical_accuracy: 0.5764
Epoch 1212/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6361 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4994 - val_sparse_categorical_accuracy: 0.5677
Epoch 1213/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6356 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5504 - val_sparse_categorical_accuracy: 0.5284
Epoch 1214/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6407 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5312 - val_sparse_categorical_accuracy: 0.5459
Epoch 1215/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6335 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5038 - val_sparse_categorical_accuracy: 0.5677
Epoch 1216/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6385 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5072 - val_sparse_categorical_accuracy: 0.5677
Epoch 1217/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6321 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5146 - val_sparse_categorical_accuracy: 0.5502
Epoch 1218/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6366 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4699 - val_sparse_categorical_accuracy: 0.5415
Epoch 1219/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6403 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4703 - val_sparse_categorical_accuracy: 0.5590
Epoch 1220/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6356 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5298 - val_sparse_categorical_accuracy: 0.5677
Epoch 1221/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6433 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5110 - val_sparse_categorical_accuracy: 0.5328
Epoch 1222/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6332 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5155 - val_sparse_categorical_accuracy: 0.5153
Epoch 1223/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6468 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5250 - val_sparse_categorical_accuracy: 0.5328
Epoch 1224/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6343 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5301 - val_sparse_categorical_accuracy: 0.5546
Epoch 1225/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6431 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.4986 - val_sparse_categorical_accuracy: 0.5852
Epoch 1226/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6428 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5326 - val_sparse_categorical_accuracy: 0.5240
Epoch 1227/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6402 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5488 - val_sparse_categorical_accuracy: 0.5721
Epoch 1228/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6393 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5231 - val_sparse_categorical_accuracy: 0.5677
Epoch 1229/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6418 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.4733 - val_sparse_categorical_accuracy: 0.5415
Epoch 1230/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6383 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4743 - val_sparse_categorical_accuracy: 0.5415
Epoch 1231/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6343 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5350 - val_sparse_categorical_accuracy: 0.5546
Epoch 1232/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6365 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5299 - val_sparse_categorical_accuracy: 0.5852
Epoch 1233/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6335 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5081 - val_sparse_categorical_accuracy: 0.5852
Epoch 1234/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6384 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5136 - val_sparse_categorical_accuracy: 0.5808
Epoch 1235/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6349 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5273 - val_sparse_categorical_accuracy: 0.5590
Epoch 1236/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6344 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4824 - val_sparse_categorical_accuracy: 0.5677
Epoch 1237/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6419 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4752 - val_sparse_categorical_accuracy: 0.5721
Epoch 1238/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6359 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5164 - val_sparse_categorical_accuracy: 0.5502
Epoch 1239/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6381 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5258 - val_sparse_categorical_accuracy: 0.5371
Epoch 1240/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6384 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5031 - val_sparse_categorical_accuracy: 0.5415
Epoch 1241/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6335 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5123 - val_sparse_categorical_accuracy: 0.5721
Epoch 1242/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6321 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5436 - val_sparse_categorical_accuracy: 0.5590
Epoch 1243/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6375 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5121 - val_sparse_categorical_accuracy: 0.5371
Epoch 1244/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6364 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4748 - val_sparse_categorical_accuracy: 0.5502
Epoch 1245/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6458 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4995 - val_sparse_categorical_accuracy: 0.5764
Epoch 1246/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6347 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5649 - val_sparse_categorical_accuracy: 0.5415
Epoch 1247/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6437 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5396 - val_sparse_categorical_accuracy: 0.5284
Epoch 1248/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6363 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5145 - val_sparse_categorical_accuracy: 0.5415
Epoch 1249/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6395 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5339 - val_sparse_categorical_accuracy: 0.5590
Epoch 1250/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6351 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5537 - val_sparse_categorical_accuracy: 0.5590
Epoch 1251/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6365 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5062 - val_sparse_categorical_accuracy: 0.5546
Epoch 1252/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6311 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4579 - val_sparse_categorical_accuracy: 0.5546
Epoch 1253/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6346 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4724 - val_sparse_categorical_accuracy: 0.5371
Epoch 1254/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6336 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5064 - val_sparse_categorical_accuracy: 0.5371
Epoch 1255/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6369 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4978 - val_sparse_categorical_accuracy: 0.5415
Epoch 1256/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6340 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4806 - val_sparse_categorical_accuracy: 0.5764
Epoch 1257/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6350 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5202 - val_sparse_categorical_accuracy: 0.5633
Epoch 1258/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6319 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5751 - val_sparse_categorical_accuracy: 0.5459
Epoch 1259/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6393 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5527 - val_sparse_categorical_accuracy: 0.5633
Epoch 1260/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6338 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5235 - val_sparse_categorical_accuracy: 0.5721
Epoch 1261/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6335 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5102 - val_sparse_categorical_accuracy: 0.5764
Epoch 1262/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6330 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5021 - val_sparse_categorical_accuracy: 0.5721
Epoch 1263/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6341 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5024 - val_sparse_categorical_accuracy: 0.5459
Epoch 1264/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6324 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5220 - val_sparse_categorical_accuracy: 0.5284
Epoch 1265/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6325 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5285 - val_sparse_categorical_accuracy: 0.5459
Epoch 1266/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6307 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5263 - val_sparse_categorical_accuracy: 0.5852
Epoch 1267/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6345 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5192 - val_sparse_categorical_accuracy: 0.5895
Epoch 1268/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6356 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5098 - val_sparse_categorical_accuracy: 0.5546
Epoch 1269/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6322 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4874 - val_sparse_categorical_accuracy: 0.5764
Epoch 1270/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6312 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4715 - val_sparse_categorical_accuracy: 0.5895
Epoch 1271/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6396 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5076 - val_sparse_categorical_accuracy: 0.5808
Epoch 1272/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6314 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5655 - val_sparse_categorical_accuracy: 0.5066
Epoch 1273/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6400 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5170 - val_sparse_categorical_accuracy: 0.5284
Epoch 1274/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6360 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.4853 - val_sparse_categorical_accuracy: 0.5808
Epoch 1275/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6340 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5127 - val_sparse_categorical_accuracy: 0.5590
Epoch 1276/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6304 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5585 - val_sparse_categorical_accuracy: 0.5328
Epoch 1277/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6356 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5215 - val_sparse_categorical_accuracy: 0.5633
Epoch 1278/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6303 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4700 - val_sparse_categorical_accuracy: 0.5852
Epoch 1279/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6338 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4592 - val_sparse_categorical_accuracy: 0.5590
Epoch 1280/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6338 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5101 - val_sparse_categorical_accuracy: 0.5328
Epoch 1281/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6316 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5620 - val_sparse_categorical_accuracy: 0.5677
Epoch 1282/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6361 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5280 - val_sparse_categorical_accuracy: 0.5764
Epoch 1283/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6387 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5279 - val_sparse_categorical_accuracy: 0.5633
Epoch 1284/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6334 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5622 - val_sparse_categorical_accuracy: 0.5328
Epoch 1285/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6373 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5286 - val_sparse_categorical_accuracy: 0.5546
Epoch 1286/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6292 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.5174 - val_sparse_categorical_accuracy: 0.5633
Epoch 1287/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6397 - sparse_categorical_accuracy: 0.7182 - val_loss: 1.5186 - val_sparse_categorical_accuracy: 0.5590
Epoch 1288/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6335 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5131 - val_sparse_categorical_accuracy: 0.5328
Epoch 1289/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6392 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5173 - val_sparse_categorical_accuracy: 0.5415
Epoch 1290/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6360 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5357 - val_sparse_categorical_accuracy: 0.5677
Epoch 1291/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6334 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5335 - val_sparse_categorical_accuracy: 0.5284
Epoch 1292/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6321 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5463 - val_sparse_categorical_accuracy: 0.5197
Epoch 1293/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6336 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5370 - val_sparse_categorical_accuracy: 0.5546
Epoch 1294/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6327 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.4934 - val_sparse_categorical_accuracy: 0.5721
Epoch 1295/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6352 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4691 - val_sparse_categorical_accuracy: 0.5633
Epoch 1296/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6402 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5276 - val_sparse_categorical_accuracy: 0.5371
Epoch 1297/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6329 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5435 - val_sparse_categorical_accuracy: 0.5459
Epoch 1298/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6347 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4834 - val_sparse_categorical_accuracy: 0.5546
Epoch 1299/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6340 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.4743 - val_sparse_categorical_accuracy: 0.5502
Epoch 1300/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6335 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5155 - val_sparse_categorical_accuracy: 0.5240
Epoch 1301/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6325 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5395 - val_sparse_categorical_accuracy: 0.5415
Epoch 1302/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6370 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5381 - val_sparse_categorical_accuracy: 0.5502
Epoch 1303/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6399 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5678 - val_sparse_categorical_accuracy: 0.5502
Epoch 1304/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6335 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5290 - val_sparse_categorical_accuracy: 0.5764
Epoch 1305/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6354 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5104 - val_sparse_categorical_accuracy: 0.5764
Epoch 1306/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6349 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5269 - val_sparse_categorical_accuracy: 0.5502
Epoch 1307/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6321 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5096 - val_sparse_categorical_accuracy: 0.5415
Epoch 1308/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6311 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5188 - val_sparse_categorical_accuracy: 0.5633
Epoch 1309/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6340 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.5573 - val_sparse_categorical_accuracy: 0.5590
Epoch 1310/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6343 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5660 - val_sparse_categorical_accuracy: 0.5153
Epoch 1311/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6345 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5404 - val_sparse_categorical_accuracy: 0.5415
Epoch 1312/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6323 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5182 - val_sparse_categorical_accuracy: 0.5721
Epoch 1313/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6358 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5209 - val_sparse_categorical_accuracy: 0.5764
Epoch 1314/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6291 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5260 - val_sparse_categorical_accuracy: 0.5328
Epoch 1315/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6343 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5101 - val_sparse_categorical_accuracy: 0.5590
Epoch 1316/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6300 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5426 - val_sparse_categorical_accuracy: 0.5895
Epoch 1317/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6371 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.5475 - val_sparse_categorical_accuracy: 0.5764
Epoch 1318/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6305 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5325 - val_sparse_categorical_accuracy: 0.5328
Epoch 1319/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6351 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5333 - val_sparse_categorical_accuracy: 0.5590
Epoch 1320/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6303 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.5310 - val_sparse_categorical_accuracy: 0.5808
Epoch 1321/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6330 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.5181 - val_sparse_categorical_accuracy: 0.5764
Epoch 1322/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6318 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5379 - val_sparse_categorical_accuracy: 0.5371
Epoch 1323/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6309 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5571 - val_sparse_categorical_accuracy: 0.5546
Epoch 1324/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6319 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5348 - val_sparse_categorical_accuracy: 0.5590
Epoch 1325/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6307 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5055 - val_sparse_categorical_accuracy: 0.5633
Epoch 1326/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6320 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5036 - val_sparse_categorical_accuracy: 0.5764
Epoch 1327/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6344 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5271 - val_sparse_categorical_accuracy: 0.5371
Epoch 1328/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6327 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5377 - val_sparse_categorical_accuracy: 0.5197
Epoch 1329/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6351 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5005 - val_sparse_categorical_accuracy: 0.5808
Epoch 1330/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6349 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5091 - val_sparse_categorical_accuracy: 0.5764
Epoch 1331/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6385 - sparse_categorical_accuracy: 0.7124 - val_loss: 1.5275 - val_sparse_categorical_accuracy: 0.5284
Epoch 1332/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6324 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5095 - val_sparse_categorical_accuracy: 0.5240
Epoch 1333/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6331 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5267 - val_sparse_categorical_accuracy: 0.5415
Epoch 1334/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6337 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5401 - val_sparse_categorical_accuracy: 0.5546
Epoch 1335/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6337 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5329 - val_sparse_categorical_accuracy: 0.5459
Epoch 1336/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6321 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5412 - val_sparse_categorical_accuracy: 0.5459
Epoch 1337/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6339 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5394 - val_sparse_categorical_accuracy: 0.5721
Epoch 1338/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6337 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.5190 - val_sparse_categorical_accuracy: 0.5721
Epoch 1339/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6329 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5262 - val_sparse_categorical_accuracy: 0.5371
Epoch 1340/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6305 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5502 - val_sparse_categorical_accuracy: 0.5240
Epoch 1341/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6306 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5426 - val_sparse_categorical_accuracy: 0.5677
Epoch 1342/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6280 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5326 - val_sparse_categorical_accuracy: 0.5633
Epoch 1343/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6286 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5516 - val_sparse_categorical_accuracy: 0.5546
Epoch 1344/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6281 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5536 - val_sparse_categorical_accuracy: 0.5590
Epoch 1345/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6282 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5181 - val_sparse_categorical_accuracy: 0.5677
Epoch 1346/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6285 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5088 - val_sparse_categorical_accuracy: 0.5764
Epoch 1347/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6313 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5352 - val_sparse_categorical_accuracy: 0.5764
Epoch 1348/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6313 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5397 - val_sparse_categorical_accuracy: 0.5371
Epoch 1349/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6310 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5168 - val_sparse_categorical_accuracy: 0.5415
Epoch 1350/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6282 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5034 - val_sparse_categorical_accuracy: 0.5764
Epoch 1351/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6298 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5371 - val_sparse_categorical_accuracy: 0.5721
Epoch 1352/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6306 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5684 - val_sparse_categorical_accuracy: 0.5764
Epoch 1353/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6303 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5339 - val_sparse_categorical_accuracy: 0.5590
Epoch 1354/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6297 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5027 - val_sparse_categorical_accuracy: 0.5721
Epoch 1355/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6303 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5258 - val_sparse_categorical_accuracy: 0.5459
Epoch 1356/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6275 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5220 - val_sparse_categorical_accuracy: 0.5328
Epoch 1357/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6286 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5382 - val_sparse_categorical_accuracy: 0.5328
Epoch 1358/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6309 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5428 - val_sparse_categorical_accuracy: 0.5328
Epoch 1359/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6328 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4963 - val_sparse_categorical_accuracy: 0.5459
Epoch 1360/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6315 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4735 - val_sparse_categorical_accuracy: 0.5459
Epoch 1361/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6472 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5180 - val_sparse_categorical_accuracy: 0.5415
Epoch 1362/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6378 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.5777 - val_sparse_categorical_accuracy: 0.5590
Epoch 1363/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6339 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5871 - val_sparse_categorical_accuracy: 0.5371
Epoch 1364/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6357 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5291 - val_sparse_categorical_accuracy: 0.5328
Epoch 1365/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6313 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.4730 - val_sparse_categorical_accuracy: 0.5546
Epoch 1366/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6339 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.4577 - val_sparse_categorical_accuracy: 0.5852
Epoch 1367/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6326 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.4985 - val_sparse_categorical_accuracy: 0.5764
Epoch 1368/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6289 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5391 - val_sparse_categorical_accuracy: 0.5721
Epoch 1369/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6291 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5400 - val_sparse_categorical_accuracy: 0.5808
Epoch 1370/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6309 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5469 - val_sparse_categorical_accuracy: 0.5677
Epoch 1371/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6306 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5419 - val_sparse_categorical_accuracy: 0.5677
Epoch 1372/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6292 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5546 - val_sparse_categorical_accuracy: 0.5677
Epoch 1373/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6273 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5566 - val_sparse_categorical_accuracy: 0.5633
Epoch 1374/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6271 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5323 - val_sparse_categorical_accuracy: 0.5371
Epoch 1375/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6266 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5219 - val_sparse_categorical_accuracy: 0.5371
Epoch 1376/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6294 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5286 - val_sparse_categorical_accuracy: 0.5459
Epoch 1377/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6275 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5511 - val_sparse_categorical_accuracy: 0.5459
Epoch 1378/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6282 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5590 - val_sparse_categorical_accuracy: 0.5328
Epoch 1379/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6294 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5774 - val_sparse_categorical_accuracy: 0.5415
Epoch 1380/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6271 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.5901 - val_sparse_categorical_accuracy: 0.5590
Epoch 1381/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6291 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5455 - val_sparse_categorical_accuracy: 0.5677
Epoch 1382/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6290 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5177 - val_sparse_categorical_accuracy: 0.5764
Epoch 1383/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6301 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5641 - val_sparse_categorical_accuracy: 0.5764
Epoch 1384/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6302 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5752 - val_sparse_categorical_accuracy: 0.5197
Epoch 1385/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6269 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5181 - val_sparse_categorical_accuracy: 0.5371
Epoch 1386/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6299 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4940 - val_sparse_categorical_accuracy: 0.5721
Epoch 1387/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6338 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5400 - val_sparse_categorical_accuracy: 0.5546
Epoch 1388/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6314 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5669 - val_sparse_categorical_accuracy: 0.5240
Epoch 1389/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6328 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5394 - val_sparse_categorical_accuracy: 0.5677
Epoch 1390/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6329 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5296 - val_sparse_categorical_accuracy: 0.5808
Epoch 1391/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6333 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5620 - val_sparse_categorical_accuracy: 0.5590
Epoch 1392/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6309 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5585 - val_sparse_categorical_accuracy: 0.5153
Epoch 1393/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6298 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5106 - val_sparse_categorical_accuracy: 0.5328
Epoch 1394/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6313 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4884 - val_sparse_categorical_accuracy: 0.5677
Epoch 1395/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6337 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5119 - val_sparse_categorical_accuracy: 0.5633
Epoch 1396/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6298 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5549 - val_sparse_categorical_accuracy: 0.5328
Epoch 1397/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6302 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5402 - val_sparse_categorical_accuracy: 0.5415
Epoch 1398/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6272 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5241 - val_sparse_categorical_accuracy: 0.5808
Epoch 1399/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6309 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5445 - val_sparse_categorical_accuracy: 0.5677
Epoch 1400/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6318 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5632 - val_sparse_categorical_accuracy: 0.5328
Epoch 1401/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6292 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.5426 - val_sparse_categorical_accuracy: 0.5328
Epoch 1402/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6286 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5315 - val_sparse_categorical_accuracy: 0.5677
Epoch 1403/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6265 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5366 - val_sparse_categorical_accuracy: 0.5808
Epoch 1404/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6281 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5360 - val_sparse_categorical_accuracy: 0.5415
Epoch 1405/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6281 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5523 - val_sparse_categorical_accuracy: 0.5371
Epoch 1406/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6280 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5669 - val_sparse_categorical_accuracy: 0.5721
Epoch 1407/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6289 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5751 - val_sparse_categorical_accuracy: 0.5502
Epoch 1408/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6330 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5689 - val_sparse_categorical_accuracy: 0.5677
Epoch 1409/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6253 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5609 - val_sparse_categorical_accuracy: 0.5371
Epoch 1410/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6302 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5431 - val_sparse_categorical_accuracy: 0.5459
Epoch 1411/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6269 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5611 - val_sparse_categorical_accuracy: 0.5677
Epoch 1412/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6321 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5739 - val_sparse_categorical_accuracy: 0.5633
Epoch 1413/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6273 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5568 - val_sparse_categorical_accuracy: 0.5328
Epoch 1414/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6347 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5568 - val_sparse_categorical_accuracy: 0.5459
Epoch 1415/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6291 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5796 - val_sparse_categorical_accuracy: 0.5808
Epoch 1416/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6302 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5536 - val_sparse_categorical_accuracy: 0.5284
Epoch 1417/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6257 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5348 - val_sparse_categorical_accuracy: 0.5371
Epoch 1418/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6308 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5386 - val_sparse_categorical_accuracy: 0.5502
Epoch 1419/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6260 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5518 - val_sparse_categorical_accuracy: 0.5677
Epoch 1420/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6265 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5430 - val_sparse_categorical_accuracy: 0.5764
Epoch 1421/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6252 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.5485 - val_sparse_categorical_accuracy: 0.5633
Epoch 1422/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6243 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5549 - val_sparse_categorical_accuracy: 0.5764
Epoch 1423/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6251 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5761 - val_sparse_categorical_accuracy: 0.5764
Epoch 1424/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6264 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5792 - val_sparse_categorical_accuracy: 0.5502
Epoch 1425/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6259 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5575 - val_sparse_categorical_accuracy: 0.5546
Epoch 1426/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6268 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5373 - val_sparse_categorical_accuracy: 0.5633
Epoch 1427/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6260 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5376 - val_sparse_categorical_accuracy: 0.5371
Epoch 1428/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6293 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5478 - val_sparse_categorical_accuracy: 0.5546
Epoch 1429/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6270 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.5374 - val_sparse_categorical_accuracy: 0.5546
Epoch 1430/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6307 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5501 - val_sparse_categorical_accuracy: 0.5677
Epoch 1431/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6324 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5991 - val_sparse_categorical_accuracy: 0.5546
Epoch 1432/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6279 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5740 - val_sparse_categorical_accuracy: 0.5764
Epoch 1433/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6274 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.4965 - val_sparse_categorical_accuracy: 0.5764
Epoch 1434/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6327 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4841 - val_sparse_categorical_accuracy: 0.5371
Epoch 1435/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6310 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4929 - val_sparse_categorical_accuracy: 0.5284
Epoch 1436/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6355 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.4865 - val_sparse_categorical_accuracy: 0.5677
Epoch 1437/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6343 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.5011 - val_sparse_categorical_accuracy: 0.5721
Epoch 1438/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6326 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.4955 - val_sparse_categorical_accuracy: 0.5721
Epoch 1439/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6289 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.4980 - val_sparse_categorical_accuracy: 0.5721
Epoch 1440/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6309 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5531 - val_sparse_categorical_accuracy: 0.5764
Epoch 1441/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6315 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5874 - val_sparse_categorical_accuracy: 0.5764
Epoch 1442/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6289 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5462 - val_sparse_categorical_accuracy: 0.5371
Epoch 1443/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6285 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5094 - val_sparse_categorical_accuracy: 0.5328
Epoch 1444/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6314 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5176 - val_sparse_categorical_accuracy: 0.5459
Epoch 1445/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6278 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5552 - val_sparse_categorical_accuracy: 0.5459
Epoch 1446/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6330 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5480 - val_sparse_categorical_accuracy: 0.5415
Epoch 1447/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6255 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5404 - val_sparse_categorical_accuracy: 0.5502
Epoch 1448/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6283 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5702 - val_sparse_categorical_accuracy: 0.5546
Epoch 1449/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6260 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5972 - val_sparse_categorical_accuracy: 0.5459
Epoch 1450/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6238 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5579 - val_sparse_categorical_accuracy: 0.5677
Epoch 1451/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6280 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5587 - val_sparse_categorical_accuracy: 0.5459
Epoch 1452/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6264 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6085 - val_sparse_categorical_accuracy: 0.5502
Epoch 1453/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6290 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5891 - val_sparse_categorical_accuracy: 0.5677
Epoch 1454/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6260 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5423 - val_sparse_categorical_accuracy: 0.5633
Epoch 1455/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6395 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5677 - val_sparse_categorical_accuracy: 0.5677
Epoch 1456/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6300 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5761 - val_sparse_categorical_accuracy: 0.5459
Epoch 1457/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6267 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5481 - val_sparse_categorical_accuracy: 0.5284
Epoch 1458/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6315 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5453 - val_sparse_categorical_accuracy: 0.5459
Epoch 1459/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6279 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5564 - val_sparse_categorical_accuracy: 0.5721
Epoch 1460/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6252 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5737 - val_sparse_categorical_accuracy: 0.5371
Epoch 1461/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6262 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5832 - val_sparse_categorical_accuracy: 0.5328
Epoch 1462/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6293 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6126 - val_sparse_categorical_accuracy: 0.5633
Epoch 1463/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6311 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5842 - val_sparse_categorical_accuracy: 0.5764
Epoch 1464/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6296 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5095 - val_sparse_categorical_accuracy: 0.5721
Epoch 1465/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6282 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.4964 - val_sparse_categorical_accuracy: 0.5633
Epoch 1466/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6290 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5366 - val_sparse_categorical_accuracy: 0.5633
Epoch 1467/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6295 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5632 - val_sparse_categorical_accuracy: 0.5240
Epoch 1468/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6257 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5892 - val_sparse_categorical_accuracy: 0.5284
Epoch 1469/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6285 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5874 - val_sparse_categorical_accuracy: 0.5459
Epoch 1470/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6269 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5711 - val_sparse_categorical_accuracy: 0.5459
Epoch 1471/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6254 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.5489 - val_sparse_categorical_accuracy: 0.5502
Epoch 1472/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6268 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5395 - val_sparse_categorical_accuracy: 0.5721
Epoch 1473/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6270 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5614 - val_sparse_categorical_accuracy: 0.5721
Epoch 1474/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6253 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6030 - val_sparse_categorical_accuracy: 0.5546
Epoch 1475/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6301 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5634 - val_sparse_categorical_accuracy: 0.5371
Epoch 1476/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6267 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5430 - val_sparse_categorical_accuracy: 0.5415
Epoch 1477/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6285 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5701 - val_sparse_categorical_accuracy: 0.5721
Epoch 1478/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6297 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5975 - val_sparse_categorical_accuracy: 0.5590
Epoch 1479/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6254 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5858 - val_sparse_categorical_accuracy: 0.5328
Epoch 1480/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6350 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5826 - val_sparse_categorical_accuracy: 0.5459
Epoch 1481/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6283 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5842 - val_sparse_categorical_accuracy: 0.5764
Epoch 1482/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6311 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5456 - val_sparse_categorical_accuracy: 0.5677
Epoch 1483/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6264 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5640 - val_sparse_categorical_accuracy: 0.5284
Epoch 1484/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6325 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5792 - val_sparse_categorical_accuracy: 0.5677
Epoch 1485/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6274 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5664 - val_sparse_categorical_accuracy: 0.5633
Epoch 1486/2000
</pre>
<pre>
2/2 [==============================] - 0s 16ms/step - loss: 0.6396 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5481 - val_sparse_categorical_accuracy: 0.5677
Epoch 1487/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6261 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.5776 - val_sparse_categorical_accuracy: 0.5633
Epoch 1488/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6260 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5767 - val_sparse_categorical_accuracy: 0.5546
Epoch 1489/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6247 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5566 - val_sparse_categorical_accuracy: 0.5721
Epoch 1490/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6227 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.5794 - val_sparse_categorical_accuracy: 0.5546
Epoch 1491/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6287 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5518 - val_sparse_categorical_accuracy: 0.5590
Epoch 1492/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6297 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5176 - val_sparse_categorical_accuracy: 0.5808
Epoch 1493/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6292 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5429 - val_sparse_categorical_accuracy: 0.5633
Epoch 1494/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6244 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5591 - val_sparse_categorical_accuracy: 0.5677
Epoch 1495/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6252 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5328 - val_sparse_categorical_accuracy: 0.5633
Epoch 1496/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6297 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5342 - val_sparse_categorical_accuracy: 0.5633
Epoch 1497/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6362 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6165 - val_sparse_categorical_accuracy: 0.5459
Epoch 1498/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6255 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6411 - val_sparse_categorical_accuracy: 0.5284
Epoch 1499/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6290 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5604 - val_sparse_categorical_accuracy: 0.5415
Epoch 1500/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6254 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5285 - val_sparse_categorical_accuracy: 0.5721
Epoch 1501/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6281 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5741 - val_sparse_categorical_accuracy: 0.5764
Epoch 1502/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6269 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5832 - val_sparse_categorical_accuracy: 0.5764
Epoch 1503/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6229 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5229 - val_sparse_categorical_accuracy: 0.5677
Epoch 1504/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6294 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5487 - val_sparse_categorical_accuracy: 0.5415
Epoch 1505/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6245 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6106 - val_sparse_categorical_accuracy: 0.5284
Epoch 1506/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6244 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6222 - val_sparse_categorical_accuracy: 0.5371
Epoch 1507/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6243 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5691 - val_sparse_categorical_accuracy: 0.5459
Epoch 1508/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6263 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5568 - val_sparse_categorical_accuracy: 0.5415
Epoch 1509/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6312 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5943 - val_sparse_categorical_accuracy: 0.5328
Epoch 1510/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6299 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5747 - val_sparse_categorical_accuracy: 0.5633
Epoch 1511/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6259 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5281 - val_sparse_categorical_accuracy: 0.5721
Epoch 1512/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6313 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5605 - val_sparse_categorical_accuracy: 0.5284
Epoch 1513/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6344 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5949 - val_sparse_categorical_accuracy: 0.5415
Epoch 1514/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6298 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5573 - val_sparse_categorical_accuracy: 0.5633
Epoch 1515/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6310 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5505 - val_sparse_categorical_accuracy: 0.5721
Epoch 1516/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6303 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5892 - val_sparse_categorical_accuracy: 0.5197
Epoch 1517/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6323 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6077 - val_sparse_categorical_accuracy: 0.5328
Epoch 1518/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6293 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5428 - val_sparse_categorical_accuracy: 0.5677
Epoch 1519/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6258 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5264 - val_sparse_categorical_accuracy: 0.5677
Epoch 1520/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6252 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5622 - val_sparse_categorical_accuracy: 0.5328
Epoch 1521/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6286 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5897 - val_sparse_categorical_accuracy: 0.5328
Epoch 1522/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6269 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5864 - val_sparse_categorical_accuracy: 0.5677
Epoch 1523/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6332 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5392 - val_sparse_categorical_accuracy: 0.5502
Epoch 1524/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6280 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.5035 - val_sparse_categorical_accuracy: 0.5371
Epoch 1525/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6369 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.4699 - val_sparse_categorical_accuracy: 0.5808
Epoch 1526/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6333 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.4635 - val_sparse_categorical_accuracy: 0.5590
Epoch 1527/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6408 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.4908 - val_sparse_categorical_accuracy: 0.5590
Epoch 1528/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6338 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5187 - val_sparse_categorical_accuracy: 0.5284
Epoch 1529/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6321 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.4999 - val_sparse_categorical_accuracy: 0.5459
Epoch 1530/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6271 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5206 - val_sparse_categorical_accuracy: 0.5852
Epoch 1531/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6319 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6205 - val_sparse_categorical_accuracy: 0.5633
Epoch 1532/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6258 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.6653 - val_sparse_categorical_accuracy: 0.5284
Epoch 1533/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6293 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6235 - val_sparse_categorical_accuracy: 0.5240
Epoch 1534/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6274 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5809 - val_sparse_categorical_accuracy: 0.5546
Epoch 1535/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6299 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5941 - val_sparse_categorical_accuracy: 0.5764
Epoch 1536/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6225 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6249 - val_sparse_categorical_accuracy: 0.5459
Epoch 1537/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6294 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6089 - val_sparse_categorical_accuracy: 0.5546
Epoch 1538/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6261 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.5855 - val_sparse_categorical_accuracy: 0.5633
Epoch 1539/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6249 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5755 - val_sparse_categorical_accuracy: 0.5677
Epoch 1540/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6226 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5969 - val_sparse_categorical_accuracy: 0.5415
Epoch 1541/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6265 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5643 - val_sparse_categorical_accuracy: 0.5371
Epoch 1542/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6290 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5362 - val_sparse_categorical_accuracy: 0.5415
Epoch 1543/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6314 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5939 - val_sparse_categorical_accuracy: 0.5415
Epoch 1544/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6283 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.6324 - val_sparse_categorical_accuracy: 0.5153
Epoch 1545/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6255 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.6022 - val_sparse_categorical_accuracy: 0.5197
Epoch 1546/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6298 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5778 - val_sparse_categorical_accuracy: 0.5328
Epoch 1547/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6257 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5878 - val_sparse_categorical_accuracy: 0.5764
Epoch 1548/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6275 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.6320 - val_sparse_categorical_accuracy: 0.5677
Epoch 1549/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6287 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.5943 - val_sparse_categorical_accuracy: 0.5371
Epoch 1550/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6311 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5472 - val_sparse_categorical_accuracy: 0.5677
Epoch 1551/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6241 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5848 - val_sparse_categorical_accuracy: 0.5546
Epoch 1552/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6398 - sparse_categorical_accuracy: 0.7197 - val_loss: 1.6052 - val_sparse_categorical_accuracy: 0.5459
Epoch 1553/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6271 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6338 - val_sparse_categorical_accuracy: 0.5066
Epoch 1554/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6371 - sparse_categorical_accuracy: 0.7241 - val_loss: 1.6009 - val_sparse_categorical_accuracy: 0.5371
Epoch 1555/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6248 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5784 - val_sparse_categorical_accuracy: 0.5808
Epoch 1556/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6295 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.6227 - val_sparse_categorical_accuracy: 0.5459
Epoch 1557/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6248 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6493 - val_sparse_categorical_accuracy: 0.5197
Epoch 1558/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6290 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5875 - val_sparse_categorical_accuracy: 0.5546
Epoch 1559/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6238 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.5594 - val_sparse_categorical_accuracy: 0.5721
Epoch 1560/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6251 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5899 - val_sparse_categorical_accuracy: 0.5415
Epoch 1561/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6227 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6080 - val_sparse_categorical_accuracy: 0.5328
Epoch 1562/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6224 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5990 - val_sparse_categorical_accuracy: 0.5677
Epoch 1563/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6249 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.5980 - val_sparse_categorical_accuracy: 0.5721
Epoch 1564/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6231 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6087 - val_sparse_categorical_accuracy: 0.5153
Epoch 1565/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6228 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5890 - val_sparse_categorical_accuracy: 0.5284
Epoch 1566/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6233 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.5723 - val_sparse_categorical_accuracy: 0.5721
Epoch 1567/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6199 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5966 - val_sparse_categorical_accuracy: 0.5677
Epoch 1568/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6209 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6154 - val_sparse_categorical_accuracy: 0.5677
Epoch 1569/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6225 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6099 - val_sparse_categorical_accuracy: 0.5677
Epoch 1570/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6215 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5957 - val_sparse_categorical_accuracy: 0.5590
Epoch 1571/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6189 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5961 - val_sparse_categorical_accuracy: 0.5502
Epoch 1572/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6195 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6115 - val_sparse_categorical_accuracy: 0.5415
Epoch 1573/2000
2/2 [==============================] - 0s 20ms/step - loss: 0.6198 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6098 - val_sparse_categorical_accuracy: 0.5328
Epoch 1574/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6193 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6092 - val_sparse_categorical_accuracy: 0.5459
Epoch 1575/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6211 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6133 - val_sparse_categorical_accuracy: 0.5721
Epoch 1576/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6207 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5840 - val_sparse_categorical_accuracy: 0.5721
Epoch 1577/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6210 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5727 - val_sparse_categorical_accuracy: 0.5721
Epoch 1578/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6254 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5849 - val_sparse_categorical_accuracy: 0.5633
Epoch 1579/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6229 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5618 - val_sparse_categorical_accuracy: 0.5808
Epoch 1580/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6208 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5827 - val_sparse_categorical_accuracy: 0.5371
Epoch 1581/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6231 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6206 - val_sparse_categorical_accuracy: 0.5240
Epoch 1582/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6224 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6051 - val_sparse_categorical_accuracy: 0.5721
Epoch 1583/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6222 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6041 - val_sparse_categorical_accuracy: 0.5677
Epoch 1584/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6204 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6127 - val_sparse_categorical_accuracy: 0.5284
Epoch 1585/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6239 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6112 - val_sparse_categorical_accuracy: 0.5284
Epoch 1586/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6216 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6141 - val_sparse_categorical_accuracy: 0.5721
Epoch 1587/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6216 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6198 - val_sparse_categorical_accuracy: 0.5633
Epoch 1588/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6220 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6214 - val_sparse_categorical_accuracy: 0.5197
Epoch 1589/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6204 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6269 - val_sparse_categorical_accuracy: 0.5066
Epoch 1590/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6280 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6203 - val_sparse_categorical_accuracy: 0.5371
Epoch 1591/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6270 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6121 - val_sparse_categorical_accuracy: 0.5546
Epoch 1592/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6209 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.6205 - val_sparse_categorical_accuracy: 0.5197
Epoch 1593/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6235 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6444 - val_sparse_categorical_accuracy: 0.5197
Epoch 1594/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6262 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6143 - val_sparse_categorical_accuracy: 0.5633
Epoch 1595/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6217 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.5487 - val_sparse_categorical_accuracy: 0.5852
Epoch 1596/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6271 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5828 - val_sparse_categorical_accuracy: 0.5459
Epoch 1597/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6221 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6557 - val_sparse_categorical_accuracy: 0.5546
Epoch 1598/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6258 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5938 - val_sparse_categorical_accuracy: 0.5633
Epoch 1599/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6214 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5636 - val_sparse_categorical_accuracy: 0.5546
Epoch 1600/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6247 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6159 - val_sparse_categorical_accuracy: 0.5328
Epoch 1601/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6213 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6259 - val_sparse_categorical_accuracy: 0.5764
Epoch 1602/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6242 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.5892 - val_sparse_categorical_accuracy: 0.5677
Epoch 1603/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6222 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6060 - val_sparse_categorical_accuracy: 0.5502
Epoch 1604/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6222 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6204 - val_sparse_categorical_accuracy: 0.5328
Epoch 1605/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6206 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5928 - val_sparse_categorical_accuracy: 0.5459
Epoch 1606/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6224 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6094 - val_sparse_categorical_accuracy: 0.5633
Epoch 1607/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6234 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6465 - val_sparse_categorical_accuracy: 0.5590
Epoch 1608/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6185 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6638 - val_sparse_categorical_accuracy: 0.5240
Epoch 1609/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6228 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6115 - val_sparse_categorical_accuracy: 0.5590
Epoch 1610/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6192 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.5825 - val_sparse_categorical_accuracy: 0.5633
Epoch 1611/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6225 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6368 - val_sparse_categorical_accuracy: 0.5546
Epoch 1612/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6199 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6694 - val_sparse_categorical_accuracy: 0.5546
Epoch 1613/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6214 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6401 - val_sparse_categorical_accuracy: 0.5546
Epoch 1614/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6201 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6169 - val_sparse_categorical_accuracy: 0.5502
Epoch 1615/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6204 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6499 - val_sparse_categorical_accuracy: 0.5284
Epoch 1616/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6227 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6265 - val_sparse_categorical_accuracy: 0.5459
Epoch 1617/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6222 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5950 - val_sparse_categorical_accuracy: 0.5546
Epoch 1618/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6226 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6321 - val_sparse_categorical_accuracy: 0.5415
Epoch 1619/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6205 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6033 - val_sparse_categorical_accuracy: 0.5502
Epoch 1620/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6205 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.5762 - val_sparse_categorical_accuracy: 0.5808
Epoch 1621/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6223 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6394 - val_sparse_categorical_accuracy: 0.5677
Epoch 1622/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6212 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6592 - val_sparse_categorical_accuracy: 0.5459
Epoch 1623/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6258 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.5946 - val_sparse_categorical_accuracy: 0.5546
Epoch 1624/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6227 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.5612 - val_sparse_categorical_accuracy: 0.5721
Epoch 1625/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6270 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6002 - val_sparse_categorical_accuracy: 0.5415
Epoch 1626/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6285 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.6109 - val_sparse_categorical_accuracy: 0.5328
Epoch 1627/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6208 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6366 - val_sparse_categorical_accuracy: 0.5415
Epoch 1628/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6281 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6213 - val_sparse_categorical_accuracy: 0.5590
Epoch 1629/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6238 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6055 - val_sparse_categorical_accuracy: 0.5240
Epoch 1630/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6210 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6084 - val_sparse_categorical_accuracy: 0.5633
Epoch 1631/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6204 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6158 - val_sparse_categorical_accuracy: 0.5633
Epoch 1632/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6215 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6270 - val_sparse_categorical_accuracy: 0.5590
Epoch 1633/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6197 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6458 - val_sparse_categorical_accuracy: 0.5590
Epoch 1634/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6199 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6336 - val_sparse_categorical_accuracy: 0.5371
Epoch 1635/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6189 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6184 - val_sparse_categorical_accuracy: 0.5459
Epoch 1636/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6219 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.6095 - val_sparse_categorical_accuracy: 0.5371
Epoch 1637/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6211 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6139 - val_sparse_categorical_accuracy: 0.5415
Epoch 1638/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6209 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6530 - val_sparse_categorical_accuracy: 0.5371
Epoch 1639/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6222 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6158 - val_sparse_categorical_accuracy: 0.5633
Epoch 1640/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6210 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.5657 - val_sparse_categorical_accuracy: 0.5764
Epoch 1641/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6206 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6031 - val_sparse_categorical_accuracy: 0.5633
Epoch 1642/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6214 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6177 - val_sparse_categorical_accuracy: 0.5721
Epoch 1643/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6196 - sparse_categorical_accuracy: 0.7533 - val_loss: 1.6083 - val_sparse_categorical_accuracy: 0.5546
Epoch 1644/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6273 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6202 - val_sparse_categorical_accuracy: 0.5677
Epoch 1645/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6234 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6310 - val_sparse_categorical_accuracy: 0.5240
Epoch 1646/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6248 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.5771 - val_sparse_categorical_accuracy: 0.5764
Epoch 1647/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6231 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.5603 - val_sparse_categorical_accuracy: 0.5852
Epoch 1648/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6221 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6411 - val_sparse_categorical_accuracy: 0.5633
Epoch 1649/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6179 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6854 - val_sparse_categorical_accuracy: 0.5197
Epoch 1650/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6268 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.6065 - val_sparse_categorical_accuracy: 0.5328
Epoch 1651/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6207 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5881 - val_sparse_categorical_accuracy: 0.5633
Epoch 1652/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6276 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6724 - val_sparse_categorical_accuracy: 0.5371
Epoch 1653/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6283 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6916 - val_sparse_categorical_accuracy: 0.5022
Epoch 1654/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6299 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6117 - val_sparse_categorical_accuracy: 0.5633
Epoch 1655/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6251 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6060 - val_sparse_categorical_accuracy: 0.5677
Epoch 1656/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6206 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6459 - val_sparse_categorical_accuracy: 0.5197
Epoch 1657/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6246 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6022 - val_sparse_categorical_accuracy: 0.5197
Epoch 1658/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6279 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.5667 - val_sparse_categorical_accuracy: 0.5502
Epoch 1659/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6231 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6399 - val_sparse_categorical_accuracy: 0.5371
Epoch 1660/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6193 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6891 - val_sparse_categorical_accuracy: 0.5153
Epoch 1661/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6261 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.6210 - val_sparse_categorical_accuracy: 0.5328
Epoch 1662/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6224 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6003 - val_sparse_categorical_accuracy: 0.5677
Epoch 1663/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6248 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6649 - val_sparse_categorical_accuracy: 0.5633
Epoch 1664/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6244 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6808 - val_sparse_categorical_accuracy: 0.5502
Epoch 1665/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6191 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6369 - val_sparse_categorical_accuracy: 0.5677
Epoch 1666/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6177 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.5975 - val_sparse_categorical_accuracy: 0.5764
Epoch 1667/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6183 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.6049 - val_sparse_categorical_accuracy: 0.5590
Epoch 1668/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6199 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6059 - val_sparse_categorical_accuracy: 0.5371
Epoch 1669/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6208 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6149 - val_sparse_categorical_accuracy: 0.5240
Epoch 1670/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6188 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6314 - val_sparse_categorical_accuracy: 0.5633
Epoch 1671/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6179 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6637 - val_sparse_categorical_accuracy: 0.5633
Epoch 1672/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6183 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6744 - val_sparse_categorical_accuracy: 0.5590
Epoch 1673/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6176 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6235 - val_sparse_categorical_accuracy: 0.5502
Epoch 1674/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6207 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.6056 - val_sparse_categorical_accuracy: 0.5808
Epoch 1675/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6202 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.6071 - val_sparse_categorical_accuracy: 0.5721
Epoch 1676/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6250 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5844 - val_sparse_categorical_accuracy: 0.5590
Epoch 1677/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6214 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6355 - val_sparse_categorical_accuracy: 0.5284
Epoch 1678/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6240 - sparse_categorical_accuracy: 0.7255 - val_loss: 1.6554 - val_sparse_categorical_accuracy: 0.5677
Epoch 1679/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6264 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5917 - val_sparse_categorical_accuracy: 0.5633
Epoch 1680/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6263 - sparse_categorical_accuracy: 0.7270 - val_loss: 1.5550 - val_sparse_categorical_accuracy: 0.5328
Epoch 1681/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6250 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5925 - val_sparse_categorical_accuracy: 0.5415
Epoch 1682/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6215 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6275 - val_sparse_categorical_accuracy: 0.5590
Epoch 1683/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6226 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.5778 - val_sparse_categorical_accuracy: 0.5764
Epoch 1684/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6296 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.5895 - val_sparse_categorical_accuracy: 0.5633
Epoch 1685/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6223 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6709 - val_sparse_categorical_accuracy: 0.5415
Epoch 1686/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6228 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6749 - val_sparse_categorical_accuracy: 0.5415
Epoch 1687/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6205 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6171 - val_sparse_categorical_accuracy: 0.5502
Epoch 1688/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6215 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6273 - val_sparse_categorical_accuracy: 0.5328
Epoch 1689/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6202 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6667 - val_sparse_categorical_accuracy: 0.5371
Epoch 1690/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6197 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6273 - val_sparse_categorical_accuracy: 0.5633
Epoch 1691/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6205 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.5772 - val_sparse_categorical_accuracy: 0.5590
Epoch 1692/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6259 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.5979 - val_sparse_categorical_accuracy: 0.5546
Epoch 1693/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6249 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6582 - val_sparse_categorical_accuracy: 0.5240
Epoch 1694/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6218 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6814 - val_sparse_categorical_accuracy: 0.4978
Epoch 1695/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6265 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6513 - val_sparse_categorical_accuracy: 0.5328
Epoch 1696/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6173 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6536 - val_sparse_categorical_accuracy: 0.5590
Epoch 1697/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6217 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6613 - val_sparse_categorical_accuracy: 0.5284
Epoch 1698/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6168 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6574 - val_sparse_categorical_accuracy: 0.5109
Epoch 1699/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6260 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6477 - val_sparse_categorical_accuracy: 0.5328
Epoch 1700/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6169 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6594 - val_sparse_categorical_accuracy: 0.5677
Epoch 1701/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6207 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6525 - val_sparse_categorical_accuracy: 0.5808
Epoch 1702/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6211 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6527 - val_sparse_categorical_accuracy: 0.5328
Epoch 1703/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6219 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6479 - val_sparse_categorical_accuracy: 0.5590
Epoch 1704/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6160 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6458 - val_sparse_categorical_accuracy: 0.5590
Epoch 1705/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6157 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6368 - val_sparse_categorical_accuracy: 0.5502
Epoch 1706/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6163 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6319 - val_sparse_categorical_accuracy: 0.5328
Epoch 1707/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6170 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6441 - val_sparse_categorical_accuracy: 0.5284
Epoch 1708/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6163 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6465 - val_sparse_categorical_accuracy: 0.5459
Epoch 1709/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6169 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6380 - val_sparse_categorical_accuracy: 0.5721
Epoch 1710/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6178 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6085 - val_sparse_categorical_accuracy: 0.5633
Epoch 1711/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6179 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.5968 - val_sparse_categorical_accuracy: 0.5633
Epoch 1712/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6182 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6208 - val_sparse_categorical_accuracy: 0.5721
Epoch 1713/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6156 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6335 - val_sparse_categorical_accuracy: 0.5546
Epoch 1714/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6170 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6717 - val_sparse_categorical_accuracy: 0.5284
Epoch 1715/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6187 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6742 - val_sparse_categorical_accuracy: 0.5546
Epoch 1716/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6156 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6419 - val_sparse_categorical_accuracy: 0.5677
Epoch 1717/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6160 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6490 - val_sparse_categorical_accuracy: 0.5415
Epoch 1718/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6185 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7077 - val_sparse_categorical_accuracy: 0.5328
Epoch 1719/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6204 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7084 - val_sparse_categorical_accuracy: 0.5371
Epoch 1720/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6174 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6612 - val_sparse_categorical_accuracy: 0.5721
Epoch 1721/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6173 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6481 - val_sparse_categorical_accuracy: 0.5764
Epoch 1722/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6175 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6615 - val_sparse_categorical_accuracy: 0.5546
Epoch 1723/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6153 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6294 - val_sparse_categorical_accuracy: 0.5677
Epoch 1724/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6155 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.6369 - val_sparse_categorical_accuracy: 0.5633
Epoch 1725/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6163 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6659 - val_sparse_categorical_accuracy: 0.5677
Epoch 1726/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6171 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6555 - val_sparse_categorical_accuracy: 0.5677
Epoch 1727/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6158 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6494 - val_sparse_categorical_accuracy: 0.5371
Epoch 1728/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6197 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6910 - val_sparse_categorical_accuracy: 0.5197
Epoch 1729/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6201 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6773 - val_sparse_categorical_accuracy: 0.5546
Epoch 1730/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6187 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6550 - val_sparse_categorical_accuracy: 0.5590
Epoch 1731/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6166 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6708 - val_sparse_categorical_accuracy: 0.5284
Epoch 1732/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6190 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6730 - val_sparse_categorical_accuracy: 0.5502
Epoch 1733/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6147 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6453 - val_sparse_categorical_accuracy: 0.5633
Epoch 1734/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6183 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6125 - val_sparse_categorical_accuracy: 0.5633
Epoch 1735/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6243 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6318 - val_sparse_categorical_accuracy: 0.5677
Epoch 1736/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6166 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6669 - val_sparse_categorical_accuracy: 0.5328
Epoch 1737/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6172 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6498 - val_sparse_categorical_accuracy: 0.5546
Epoch 1738/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6145 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6364 - val_sparse_categorical_accuracy: 0.5677
Epoch 1739/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6185 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6382 - val_sparse_categorical_accuracy: 0.5546
Epoch 1740/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6155 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6498 - val_sparse_categorical_accuracy: 0.5284
Epoch 1741/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6161 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6698 - val_sparse_categorical_accuracy: 0.5328
Epoch 1742/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6151 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6694 - val_sparse_categorical_accuracy: 0.5808
Epoch 1743/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6170 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6610 - val_sparse_categorical_accuracy: 0.5633
Epoch 1744/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6143 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6512 - val_sparse_categorical_accuracy: 0.5284
Epoch 1745/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6207 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6305 - val_sparse_categorical_accuracy: 0.5328
Epoch 1746/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6167 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6240 - val_sparse_categorical_accuracy: 0.5633
Epoch 1747/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6203 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6374 - val_sparse_categorical_accuracy: 0.5633
Epoch 1748/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6225 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6854 - val_sparse_categorical_accuracy: 0.5153
Epoch 1749/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6157 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6858 - val_sparse_categorical_accuracy: 0.5677
Epoch 1750/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6179 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6512 - val_sparse_categorical_accuracy: 0.5764
Epoch 1751/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6195 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.6594 - val_sparse_categorical_accuracy: 0.5371
Epoch 1752/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6173 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7017 - val_sparse_categorical_accuracy: 0.5459
Epoch 1753/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6159 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6828 - val_sparse_categorical_accuracy: 0.5633
Epoch 1754/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6178 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6372 - val_sparse_categorical_accuracy: 0.5852
Epoch 1755/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6149 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6678 - val_sparse_categorical_accuracy: 0.5415
Epoch 1756/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6158 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6821 - val_sparse_categorical_accuracy: 0.5197
Epoch 1757/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6172 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6686 - val_sparse_categorical_accuracy: 0.5284
Epoch 1758/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6156 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6627 - val_sparse_categorical_accuracy: 0.5459
Epoch 1759/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6157 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6687 - val_sparse_categorical_accuracy: 0.5459
Epoch 1760/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6144 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6598 - val_sparse_categorical_accuracy: 0.5371
Epoch 1761/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6157 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6701 - val_sparse_categorical_accuracy: 0.5371
Epoch 1762/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6159 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6832 - val_sparse_categorical_accuracy: 0.5546
Epoch 1763/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6121 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6721 - val_sparse_categorical_accuracy: 0.5590
Epoch 1764/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6147 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6692 - val_sparse_categorical_accuracy: 0.5415
Epoch 1765/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6154 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6744 - val_sparse_categorical_accuracy: 0.5546
Epoch 1766/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6169 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6495 - val_sparse_categorical_accuracy: 0.5459
Epoch 1767/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6172 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6443 - val_sparse_categorical_accuracy: 0.5284
Epoch 1768/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6205 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6414 - val_sparse_categorical_accuracy: 0.5677
Epoch 1769/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6145 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6459 - val_sparse_categorical_accuracy: 0.5590
Epoch 1770/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6256 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6231 - val_sparse_categorical_accuracy: 0.5502
Epoch 1771/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6175 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6516 - val_sparse_categorical_accuracy: 0.5240
Epoch 1772/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6203 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6881 - val_sparse_categorical_accuracy: 0.5633
Epoch 1773/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6150 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6933 - val_sparse_categorical_accuracy: 0.5764
Epoch 1774/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6162 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6887 - val_sparse_categorical_accuracy: 0.5590
Epoch 1775/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6141 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.6832 - val_sparse_categorical_accuracy: 0.5153
Epoch 1776/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6231 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6476 - val_sparse_categorical_accuracy: 0.5371
Epoch 1777/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6184 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6612 - val_sparse_categorical_accuracy: 0.5590
Epoch 1778/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6143 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6860 - val_sparse_categorical_accuracy: 0.5633
Epoch 1779/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6157 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6796 - val_sparse_categorical_accuracy: 0.5371
Epoch 1780/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6165 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6860 - val_sparse_categorical_accuracy: 0.5371
Epoch 1781/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6143 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6815 - val_sparse_categorical_accuracy: 0.5633
Epoch 1782/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6167 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6776 - val_sparse_categorical_accuracy: 0.5633
Epoch 1783/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6140 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6699 - val_sparse_categorical_accuracy: 0.5371
Epoch 1784/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6164 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6836 - val_sparse_categorical_accuracy: 0.5284
Epoch 1785/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6179 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7021 - val_sparse_categorical_accuracy: 0.5677
Epoch 1786/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6161 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6467 - val_sparse_categorical_accuracy: 0.5677
Epoch 1787/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6159 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6374 - val_sparse_categorical_accuracy: 0.5633
Epoch 1788/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6143 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6811 - val_sparse_categorical_accuracy: 0.5153
Epoch 1789/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6211 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6367 - val_sparse_categorical_accuracy: 0.5546
Epoch 1790/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6177 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.5997 - val_sparse_categorical_accuracy: 0.5721
Epoch 1791/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6218 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6732 - val_sparse_categorical_accuracy: 0.5415
Epoch 1792/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6182 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6845 - val_sparse_categorical_accuracy: 0.5197
Epoch 1793/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6170 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6407 - val_sparse_categorical_accuracy: 0.5371
Epoch 1794/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6271 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6766 - val_sparse_categorical_accuracy: 0.5240
Epoch 1795/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6179 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.7297 - val_sparse_categorical_accuracy: 0.5240
Epoch 1796/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6189 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6696 - val_sparse_categorical_accuracy: 0.5371
Epoch 1797/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6159 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6174 - val_sparse_categorical_accuracy: 0.5764
Epoch 1798/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6158 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6199 - val_sparse_categorical_accuracy: 0.5721
Epoch 1799/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6190 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6405 - val_sparse_categorical_accuracy: 0.5677
Epoch 1800/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6180 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6429 - val_sparse_categorical_accuracy: 0.5590
Epoch 1801/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6139 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7038 - val_sparse_categorical_accuracy: 0.5240
Epoch 1802/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6196 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7349 - val_sparse_categorical_accuracy: 0.5153
Epoch 1803/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6175 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6793 - val_sparse_categorical_accuracy: 0.5415
Epoch 1804/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6169 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6352 - val_sparse_categorical_accuracy: 0.5284
Epoch 1805/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6164 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6747 - val_sparse_categorical_accuracy: 0.5197
Epoch 1806/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6174 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6934 - val_sparse_categorical_accuracy: 0.5328
Epoch 1807/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6153 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6576 - val_sparse_categorical_accuracy: 0.5590
Epoch 1808/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6200 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6690 - val_sparse_categorical_accuracy: 0.5546
Epoch 1809/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6130 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7143 - val_sparse_categorical_accuracy: 0.5284
Epoch 1810/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6230 - sparse_categorical_accuracy: 0.7226 - val_loss: 1.6827 - val_sparse_categorical_accuracy: 0.5546
Epoch 1811/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6182 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6692 - val_sparse_categorical_accuracy: 0.5590
Epoch 1812/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6159 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7155 - val_sparse_categorical_accuracy: 0.5546
Epoch 1813/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6187 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6982 - val_sparse_categorical_accuracy: 0.5371
Epoch 1814/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6128 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6421 - val_sparse_categorical_accuracy: 0.5633
Epoch 1815/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6203 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6695 - val_sparse_categorical_accuracy: 0.5459
Epoch 1816/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6118 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7332 - val_sparse_categorical_accuracy: 0.5240
Epoch 1817/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6169 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7078 - val_sparse_categorical_accuracy: 0.5284
Epoch 1818/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6128 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6691 - val_sparse_categorical_accuracy: 0.5590
Epoch 1819/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6155 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6926 - val_sparse_categorical_accuracy: 0.5459
Epoch 1820/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6138 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.7165 - val_sparse_categorical_accuracy: 0.5197
Epoch 1821/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6168 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6693 - val_sparse_categorical_accuracy: 0.5633
Epoch 1822/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6144 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6817 - val_sparse_categorical_accuracy: 0.5590
Epoch 1823/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6171 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.7042 - val_sparse_categorical_accuracy: 0.5371
Epoch 1824/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6131 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6959 - val_sparse_categorical_accuracy: 0.5546
Epoch 1825/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6146 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7100 - val_sparse_categorical_accuracy: 0.5502
Epoch 1826/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6180 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6883 - val_sparse_categorical_accuracy: 0.5459
Epoch 1827/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6148 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6571 - val_sparse_categorical_accuracy: 0.5677
Epoch 1828/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6128 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6750 - val_sparse_categorical_accuracy: 0.5371
Epoch 1829/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6150 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6846 - val_sparse_categorical_accuracy: 0.5284
Epoch 1830/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6160 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6784 - val_sparse_categorical_accuracy: 0.5459
Epoch 1831/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6146 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.6691 - val_sparse_categorical_accuracy: 0.5502
Epoch 1832/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6201 - sparse_categorical_accuracy: 0.7212 - val_loss: 1.6492 - val_sparse_categorical_accuracy: 0.5284
Epoch 1833/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6183 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6540 - val_sparse_categorical_accuracy: 0.5415
Epoch 1834/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6202 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7148 - val_sparse_categorical_accuracy: 0.5633
Epoch 1835/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6184 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6932 - val_sparse_categorical_accuracy: 0.5240
Epoch 1836/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6198 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6411 - val_sparse_categorical_accuracy: 0.5415
Epoch 1837/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6312 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.6854 - val_sparse_categorical_accuracy: 0.5502
Epoch 1838/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6233 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7235 - val_sparse_categorical_accuracy: 0.5633
Epoch 1839/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6172 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6581 - val_sparse_categorical_accuracy: 0.5502
Epoch 1840/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6192 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6286 - val_sparse_categorical_accuracy: 0.5502
Epoch 1841/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6184 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6906 - val_sparse_categorical_accuracy: 0.5328
Epoch 1842/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6210 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6933 - val_sparse_categorical_accuracy: 0.5240
Epoch 1843/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6169 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6669 - val_sparse_categorical_accuracy: 0.5502
Epoch 1844/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6205 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.7290 - val_sparse_categorical_accuracy: 0.5590
Epoch 1845/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6224 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6987 - val_sparse_categorical_accuracy: 0.5633
Epoch 1846/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6118 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6086 - val_sparse_categorical_accuracy: 0.5502
Epoch 1847/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6338 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6379 - val_sparse_categorical_accuracy: 0.5677
Epoch 1848/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6213 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.7462 - val_sparse_categorical_accuracy: 0.5677
Epoch 1849/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6215 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6961 - val_sparse_categorical_accuracy: 0.5371
Epoch 1850/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6208 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6399 - val_sparse_categorical_accuracy: 0.5328
Epoch 1851/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6174 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6521 - val_sparse_categorical_accuracy: 0.5546
Epoch 1852/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6191 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6637 - val_sparse_categorical_accuracy: 0.5415
Epoch 1853/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6183 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.6247 - val_sparse_categorical_accuracy: 0.5459
Epoch 1854/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6196 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6675 - val_sparse_categorical_accuracy: 0.5371
Epoch 1855/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6163 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7532 - val_sparse_categorical_accuracy: 0.5415
Epoch 1856/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6157 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7129 - val_sparse_categorical_accuracy: 0.5546
Epoch 1857/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6138 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6550 - val_sparse_categorical_accuracy: 0.5459
Epoch 1858/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6188 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7215 - val_sparse_categorical_accuracy: 0.5153
Epoch 1859/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6149 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7672 - val_sparse_categorical_accuracy: 0.5284
Epoch 1860/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6211 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6725 - val_sparse_categorical_accuracy: 0.5590
Epoch 1861/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6216 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6388 - val_sparse_categorical_accuracy: 0.5590
Epoch 1862/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6302 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.7141 - val_sparse_categorical_accuracy: 0.5371
Epoch 1863/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6186 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.7555 - val_sparse_categorical_accuracy: 0.5459
Epoch 1864/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6194 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6585 - val_sparse_categorical_accuracy: 0.5721
Epoch 1865/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6160 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6388 - val_sparse_categorical_accuracy: 0.5502
Epoch 1866/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6157 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6982 - val_sparse_categorical_accuracy: 0.5459
Epoch 1867/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6145 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.7266 - val_sparse_categorical_accuracy: 0.5502
Epoch 1868/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6171 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6863 - val_sparse_categorical_accuracy: 0.5328
Epoch 1869/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6130 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6534 - val_sparse_categorical_accuracy: 0.5415
Epoch 1870/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6213 - sparse_categorical_accuracy: 0.7285 - val_loss: 1.6761 - val_sparse_categorical_accuracy: 0.5721
Epoch 1871/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6154 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.7239 - val_sparse_categorical_accuracy: 0.5633
Epoch 1872/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6226 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6797 - val_sparse_categorical_accuracy: 0.5197
Epoch 1873/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6169 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6383 - val_sparse_categorical_accuracy: 0.5197
Epoch 1874/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6257 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6522 - val_sparse_categorical_accuracy: 0.5546
Epoch 1875/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6197 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.7098 - val_sparse_categorical_accuracy: 0.5677
Epoch 1876/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6239 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.6833 - val_sparse_categorical_accuracy: 0.5371
Epoch 1877/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6193 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6649 - val_sparse_categorical_accuracy: 0.5153
Epoch 1878/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6213 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6544 - val_sparse_categorical_accuracy: 0.5721
Epoch 1879/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6255 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6394 - val_sparse_categorical_accuracy: 0.5852
Epoch 1880/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6188 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6625 - val_sparse_categorical_accuracy: 0.5284
Epoch 1881/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6207 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6947 - val_sparse_categorical_accuracy: 0.5153
Epoch 1882/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6216 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7033 - val_sparse_categorical_accuracy: 0.5502
Epoch 1883/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6144 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7448 - val_sparse_categorical_accuracy: 0.5633
Epoch 1884/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6159 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7307 - val_sparse_categorical_accuracy: 0.5284
Epoch 1885/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6137 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6914 - val_sparse_categorical_accuracy: 0.5371
Epoch 1886/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6161 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6920 - val_sparse_categorical_accuracy: 0.5415
Epoch 1887/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6129 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6946 - val_sparse_categorical_accuracy: 0.5808
Epoch 1888/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6142 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.6748 - val_sparse_categorical_accuracy: 0.5590
Epoch 1889/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6113 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6729 - val_sparse_categorical_accuracy: 0.5371
Epoch 1890/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6151 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6813 - val_sparse_categorical_accuracy: 0.5328
Epoch 1891/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6134 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6895 - val_sparse_categorical_accuracy: 0.5677
Epoch 1892/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6100 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7067 - val_sparse_categorical_accuracy: 0.5546
Epoch 1893/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6103 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7161 - val_sparse_categorical_accuracy: 0.5415
Epoch 1894/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6118 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.7239 - val_sparse_categorical_accuracy: 0.5371
Epoch 1895/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6107 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.7039 - val_sparse_categorical_accuracy: 0.5633
Epoch 1896/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6095 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7005 - val_sparse_categorical_accuracy: 0.5677
Epoch 1897/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6101 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7043 - val_sparse_categorical_accuracy: 0.5633
Epoch 1898/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6088 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6905 - val_sparse_categorical_accuracy: 0.5546
Epoch 1899/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6122 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6844 - val_sparse_categorical_accuracy: 0.5546
Epoch 1900/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6121 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7129 - val_sparse_categorical_accuracy: 0.5546
Epoch 1901/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6124 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6995 - val_sparse_categorical_accuracy: 0.5721
Epoch 1902/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6144 - sparse_categorical_accuracy: 0.7299 - val_loss: 1.6807 - val_sparse_categorical_accuracy: 0.5546
Epoch 1903/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6105 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6952 - val_sparse_categorical_accuracy: 0.5633
Epoch 1904/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6121 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6884 - val_sparse_categorical_accuracy: 0.5590
Epoch 1905/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6146 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6652 - val_sparse_categorical_accuracy: 0.5677
Epoch 1906/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6137 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6918 - val_sparse_categorical_accuracy: 0.5721
Epoch 1907/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6096 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7052 - val_sparse_categorical_accuracy: 0.5590
Epoch 1908/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6121 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7171 - val_sparse_categorical_accuracy: 0.5502
Epoch 1909/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6089 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7418 - val_sparse_categorical_accuracy: 0.5371
Epoch 1910/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6139 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6656 - val_sparse_categorical_accuracy: 0.5764
Epoch 1911/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6119 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6189 - val_sparse_categorical_accuracy: 0.5677
Epoch 1912/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6186 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6839 - val_sparse_categorical_accuracy: 0.5502
Epoch 1913/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6123 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7388 - val_sparse_categorical_accuracy: 0.5371
Epoch 1914/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6173 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6841 - val_sparse_categorical_accuracy: 0.5240
Epoch 1915/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6151 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6618 - val_sparse_categorical_accuracy: 0.5590
Epoch 1916/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6136 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7240 - val_sparse_categorical_accuracy: 0.5590
Epoch 1917/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6148 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7337 - val_sparse_categorical_accuracy: 0.5546
Epoch 1918/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6155 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.7076 - val_sparse_categorical_accuracy: 0.5546
Epoch 1919/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6106 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7038 - val_sparse_categorical_accuracy: 0.5721
Epoch 1920/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6183 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6924 - val_sparse_categorical_accuracy: 0.5677
Epoch 1921/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6117 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6585 - val_sparse_categorical_accuracy: 0.5240
Epoch 1922/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6211 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.6353 - val_sparse_categorical_accuracy: 0.5284
Epoch 1923/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6172 - sparse_categorical_accuracy: 0.7314 - val_loss: 1.6841 - val_sparse_categorical_accuracy: 0.5590
Epoch 1924/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6171 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7196 - val_sparse_categorical_accuracy: 0.5459
Epoch 1925/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6105 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7114 - val_sparse_categorical_accuracy: 0.5459
Epoch 1926/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6091 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6814 - val_sparse_categorical_accuracy: 0.5633
Epoch 1927/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6108 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6823 - val_sparse_categorical_accuracy: 0.5764
Epoch 1928/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6097 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.7000 - val_sparse_categorical_accuracy: 0.5459
Epoch 1929/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6098 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7044 - val_sparse_categorical_accuracy: 0.5502
Epoch 1930/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6098 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7015 - val_sparse_categorical_accuracy: 0.5677
Epoch 1931/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6091 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7127 - val_sparse_categorical_accuracy: 0.5459
Epoch 1932/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6117 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7036 - val_sparse_categorical_accuracy: 0.5415
Epoch 1933/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6114 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6731 - val_sparse_categorical_accuracy: 0.5677
Epoch 1934/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6092 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6885 - val_sparse_categorical_accuracy: 0.5502
Epoch 1935/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6088 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.6965 - val_sparse_categorical_accuracy: 0.5546
Epoch 1936/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6072 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7000 - val_sparse_categorical_accuracy: 0.5721
Epoch 1937/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6087 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.6945 - val_sparse_categorical_accuracy: 0.5677
Epoch 1938/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6096 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.6969 - val_sparse_categorical_accuracy: 0.5546
Epoch 1939/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6098 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7292 - val_sparse_categorical_accuracy: 0.5197
Epoch 1940/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6106 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.7202 - val_sparse_categorical_accuracy: 0.5240
Epoch 1941/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6088 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6854 - val_sparse_categorical_accuracy: 0.5328
Epoch 1942/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6119 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7099 - val_sparse_categorical_accuracy: 0.5721
Epoch 1943/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6089 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7781 - val_sparse_categorical_accuracy: 0.5328
Epoch 1944/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6137 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7422 - val_sparse_categorical_accuracy: 0.5240
Epoch 1945/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6135 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6834 - val_sparse_categorical_accuracy: 0.5721
Epoch 1946/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6142 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7075 - val_sparse_categorical_accuracy: 0.5721
Epoch 1947/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6149 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7499 - val_sparse_categorical_accuracy: 0.5633
Epoch 1948/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6087 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7273 - val_sparse_categorical_accuracy: 0.5197
Epoch 1949/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6156 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7233 - val_sparse_categorical_accuracy: 0.5328
Epoch 1950/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6104 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7259 - val_sparse_categorical_accuracy: 0.5415
Epoch 1951/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6123 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.6940 - val_sparse_categorical_accuracy: 0.5502
Epoch 1952/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6102 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6856 - val_sparse_categorical_accuracy: 0.5371
Epoch 1953/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6167 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7053 - val_sparse_categorical_accuracy: 0.5590
Epoch 1954/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6077 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.7482 - val_sparse_categorical_accuracy: 0.5764
Epoch 1955/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6179 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7371 - val_sparse_categorical_accuracy: 0.5546
Epoch 1956/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6100 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.7061 - val_sparse_categorical_accuracy: 0.5109
Epoch 1957/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6203 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7003 - val_sparse_categorical_accuracy: 0.5240
Epoch 1958/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6097 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7405 - val_sparse_categorical_accuracy: 0.5415
Epoch 1959/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6182 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.7198 - val_sparse_categorical_accuracy: 0.5415
Epoch 1960/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6093 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6912 - val_sparse_categorical_accuracy: 0.5197
Epoch 1961/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6189 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6785 - val_sparse_categorical_accuracy: 0.5502
Epoch 1962/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6135 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7275 - val_sparse_categorical_accuracy: 0.5764
Epoch 1963/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6136 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.7371 - val_sparse_categorical_accuracy: 0.5371
Epoch 1964/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6096 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.7176 - val_sparse_categorical_accuracy: 0.5371
Epoch 1965/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6133 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.6929 - val_sparse_categorical_accuracy: 0.5852
Epoch 1966/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6100 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.6852 - val_sparse_categorical_accuracy: 0.5633
Epoch 1967/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6143 - sparse_categorical_accuracy: 0.7328 - val_loss: 1.7127 - val_sparse_categorical_accuracy: 0.5371
Epoch 1968/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6087 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7281 - val_sparse_categorical_accuracy: 0.5153
Epoch 1969/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6189 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7090 - val_sparse_categorical_accuracy: 0.5371
Epoch 1970/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6125 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.7220 - val_sparse_categorical_accuracy: 0.5677
Epoch 1971/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6108 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7288 - val_sparse_categorical_accuracy: 0.5240
Epoch 1972/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6091 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7202 - val_sparse_categorical_accuracy: 0.5197
Epoch 1973/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6151 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.6963 - val_sparse_categorical_accuracy: 0.5721
Epoch 1974/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6069 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.7231 - val_sparse_categorical_accuracy: 0.5764
Epoch 1975/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6117 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.7194 - val_sparse_categorical_accuracy: 0.5633
Epoch 1976/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6058 - sparse_categorical_accuracy: 0.7533 - val_loss: 1.7174 - val_sparse_categorical_accuracy: 0.5371
Epoch 1977/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6124 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7281 - val_sparse_categorical_accuracy: 0.5371
Epoch 1978/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6100 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7271 - val_sparse_categorical_accuracy: 0.5677
Epoch 1979/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6069 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7144 - val_sparse_categorical_accuracy: 0.5415
Epoch 1980/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6068 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7132 - val_sparse_categorical_accuracy: 0.5415
Epoch 1981/2000
</pre>
<pre>
2/2 [==============================] - 0s 15ms/step - loss: 0.6061 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7295 - val_sparse_categorical_accuracy: 0.5633
Epoch 1982/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6077 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7058 - val_sparse_categorical_accuracy: 0.5590
Epoch 1983/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6078 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7201 - val_sparse_categorical_accuracy: 0.5546
Epoch 1984/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6056 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7438 - val_sparse_categorical_accuracy: 0.5546
Epoch 1985/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6071 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7160 - val_sparse_categorical_accuracy: 0.5459
Epoch 1986/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6079 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.6804 - val_sparse_categorical_accuracy: 0.5371
Epoch 1987/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6077 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.6745 - val_sparse_categorical_accuracy: 0.5590
Epoch 1988/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6097 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6789 - val_sparse_categorical_accuracy: 0.5371
Epoch 1989/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6104 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6821 - val_sparse_categorical_accuracy: 0.5459
Epoch 1990/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6102 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7252 - val_sparse_categorical_accuracy: 0.5764
Epoch 1991/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6090 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7185 - val_sparse_categorical_accuracy: 0.5764
Epoch 1992/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6073 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.6902 - val_sparse_categorical_accuracy: 0.5459
Epoch 1993/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6085 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7046 - val_sparse_categorical_accuracy: 0.5328
Epoch 1994/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6086 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7223 - val_sparse_categorical_accuracy: 0.5459
Epoch 1995/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6066 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7198 - val_sparse_categorical_accuracy: 0.5677
Epoch 1996/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6070 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.7191 - val_sparse_categorical_accuracy: 0.5459
Epoch 1997/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6140 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.7395 - val_sparse_categorical_accuracy: 0.5328
Epoch 1998/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6141 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7662 - val_sparse_categorical_accuracy: 0.5590
Epoch 1999/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6123 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7279 - val_sparse_categorical_accuracy: 0.5677
Epoch 2000/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6062 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.7147 - val_sparse_categorical_accuracy: 0.5502
</pre>

```python
hist_df = pd.DataFrame(history.history)
hist_df
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
      <th>loss</th>
      <th>sparse_categorical_accuracy</th>
      <th>val_loss</th>
      <th>val_sparse_categorical_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.683955</td>
      <td>0.713869</td>
      <td>1.223650</td>
      <td>0.567686</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.680965</td>
      <td>0.712409</td>
      <td>1.229555</td>
      <td>0.567686</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.685888</td>
      <td>0.703650</td>
      <td>1.217334</td>
      <td>0.563319</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.684157</td>
      <td>0.708029</td>
      <td>1.214830</td>
      <td>0.563319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.686004</td>
      <td>0.708029</td>
      <td>1.221601</td>
      <td>0.550218</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>0.607001</td>
      <td>0.745985</td>
      <td>1.719105</td>
      <td>0.545852</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>0.613979</td>
      <td>0.745985</td>
      <td>1.739488</td>
      <td>0.532751</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>0.614102</td>
      <td>0.744526</td>
      <td>1.766184</td>
      <td>0.558952</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>0.612287</td>
      <td>0.741606</td>
      <td>1.727893</td>
      <td>0.567686</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>0.606213</td>
      <td>0.748905</td>
      <td>1.714689</td>
      <td>0.550218</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 4 columns</p>
</div>



```python
y_vloss = hist_df['val_loss']
y_loss = hist_df["loss"]
```


```python
x_len = np.arange(len(y_loss))
```


```python
plt.plot(x_len, y_vloss, c = 'red',markersize = 2, label = 'Testset Loss')
plt.plot(x_len, y_loss, c = 'blue', markersize = 2, label = 'Trainset Loss')
plt.legend(loc = 'upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEGCAYAAABo25JHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA3U0lEQVR4nO3dd5hU5fXA8e+BpReBBVRAuoBIWWQNQWxIBLtGiQqighpCYkeMKIKoMaJgsIuNIDb8gSAIGImoYEEEDCi9l5XeO+wu5/fHO9cpOzNbZ2Zhzud55tk7t8y8exfuufct5xVVxRhjTPIqkegCGGOMSSwLBMYYk+QsEBhjTJKzQGCMMUnOAoExxiS5lEQXIL+qV6+u9evXT3QxjDHmuDJv3rztqloj3LbjLhDUr1+fuXPnJroYxhhzXBGRdZG2WdWQMcYkuZgFAhEZKSJbRWRhhO0nicinIrJARBaJSK9YlcUYY0xksXwiGAVcEmX7ncBiVW0NXAg8JyKlY1geY4wxYcSsjUBVZ4pI/Wi7AJVERICKwE4gK1blMcbETmZmJhkZGRw+fDjRRUl6ZcuWpU6dOpQqVSrPxySysfhlYBKwEagE3KCqxxJYHmNMAWVkZFCpUiXq16+Pu7cziaCq7Nixg4yMDBo0aJDn4xLZWNwFmA/UAtKAl0WkcrgdRaS3iMwVkbnbtm2LXwmNMXly+PBhUlNTLQgkmIiQmpqa7yezRAaCXsB4dVYCa4Bm4XZU1TdUNV1V02vUCNsN1hiTYBYEioeC/B0SGQjWA50ARORkoCmwOoHlMcaYxDh0CPbtS9jXx7L76IfALKCpiGSIyO0i0kdE+vh2eRI4R0R+AaYDD6nq9liVxxhz4tqxYwdpaWmkpaVxyimnULt27d/eHz16NNfjv/76a77//vsCfffatWv54IMPIm5r0aJF7h+yaBEsW1ag7y8Ksew11C2X7RuBzrH6fmNM8khNTWX+/PkADB48mIoVK9KvX788H//1119TsWJFzjnnnHx/txcIunfv7lYcOQI7dsCpp+b7syJSdZ+ZmgoxqIKzkcXGmBPSvHnzuOCCC2jbti1dunRh06ZNALz44os0b96cVq1aceONN7J27VpGjBjB8OHDSUtL45tvvmHs2LG0aNGC1q1bc/755wOQnZ3Ngw8+yNlnn02rVq14/fXXAejfvz/ffPMNaWlpDB8+HFasgI0bIcqTyPTp02nTpg0tW7bktttu44hv3/79+/9WNi+QjR07lhZnnEHrc87h/AIEqrw47nINGWOKufvuA9/deZFJS4Pnn8/z7qrK3XffzcSJE6lRowYfffQRAwYMYOTIkQwZMoQ1a9ZQpkwZdu/eTZUqVejTp0/QU0TLli35/PPPqV27Nrt37wbg7bff5qSTTmLOnDkcOXKEDh060LlzZ4YMGcKwYcOYPHmy+3IvF9revWHLdvjwYXr27Mn06dNp0qQJt9xyC6+NG8ctl1/OhAkTWLp0KSLy2/c+8cQTfP7uu9QWYXelSgU4ebmzJwJjzAnnyJEjLFy4kIsvvpi0tDT+8Y9/kJGRAUCrVq246aabeO+990hJCX8v3KFDB3r27Mmbb75JdnY2ANOmTWP06NGkpaXRrl07duzYwYoVK4IPzAoYE7sufI63ZcuW0aBBA5o0aQLArbfeysz//Y/KFSpQNiWFO267jfHjx1O+fHl/Wfr25c0JE34rS1GzJwJjTNHKx517rKgqZ555JrNmzcqxbcqUKcycOZNJkybx5JNPsmjRohz7jBgxgtmzZzNlyhTS0tKYP38+qspLL71Ely5dgvb9+uuv3UJmJixYkLMwmZmQkQF16riyHT7s6vxDpKSk8OObbzJ93jzGfPIJL7/8Ml9+9hkjBgxg9k8/MeXTT0m7+GLm//wzqamp+T8pUdgTgTHmhFOmTBm2bdv2WyDIzMxk0aJFHDt2jA0bNtCxY0eeffZZdu/ezf79+6lUqRL7Arpvrlq1inbt2vHEE09QvXp1NmzYQJcuXXjttdfIPHAAVFm+fDkHDhzwH3vkSPjCZGXB5s2/vW2WlcXaFStY6QtA7777LhecdRb7Dx5kz/79XNa+Pc8//LBr/F61ilXz5tGuWTOe6NOH6tWqsWHDhiI/X/ZEYIw54ZQoUYJx48Zxzz33sGfPHrKysrjvvvto0qQJPXr0YM+ePagq999/P1WqVOHKK6+ka9euTJw4kZdeeonhw4ezYsUKVJVOnTrRunVrWrVqxdolSzirdWs0JYUalSvzyYcf0qpVK1JSUmjdvj09L76Y+73eQz7L1q2jzuWXgy/3z/C77uLfgwbxp2uvJat0ac4++2z6XHcdO/fu5eoHHuDw0aOoKsOHDYPDh3nwhRdY8euvaHY2nS66iNatWxf5+RIN84hSnKWnp6tNTGNM8bJkyRLOOOOMRBcjtvbt8/f1r1gR9u93y+np7q4/XAN5kyawfLlbbtYMypeHn37yb09Ph927YeXKvJWhTh045ZRcdwv39xCReaqaHm5/eyIwxpi8iFT1A7Aw7LQr/iAAsHQpnHxy8PatW+FYPnJtxujG3doIjDEmv7ynAYCDB4N7C0WzZUvw+/XrI3YzDSs/QSMfLBAYY5LX5s1Q2IzGixcX7vj8BIIyZQr3XRFY1ZAxJnn5xhYQmNV47153512lin/d7t0uxUMiVakC1avH5KPticAYYwItX+5vvD12zFX7rFyZ0OygQExyDHksEBhjTjyrVrkePgcO5G3/FSvC17+vW1f06TLywzf6GLBAYIwx0QSloT75ZGq3b0/alVeS1qZNcBpq7w4/wNzFi7ln4EAXNAIDx7ZtsHNngcoz6tNP2Rih7aHn4MGMmz499w+pWtV1U/XUrl2gsuSFtREYY457QWmo//Y3KpYqRb+bb3Yb9+4lq0oVUkqWdFU8e/e6/vs+6c2bk968uQsSgQ23EXIF5cWoyZNp0agRtfI7o2LNmq5LKUDdusFPATFqKAYLBMaY41l2NpQoEbbapOfgwVQ76ST+t2wZZ7Vpww0XXMB9zz7LoSNHKJeayr+ff56mJUrw9bx5DHvvPSaPHMngl19m/Zo1rP71V9Zv3sx93bpxz403cuDQIa5/+GEytm4lOzubgbffzg2dOzNvyRL6Dh/O/kOHqF6lCqMee4zvFixg7pIl3DRwIOXKlGHWyJGUK1s26q9xuHx5/jp0KHPnziVFlX89/TQd09NZtGgRvW69laNZWRwrU4aPP/6YWrVqcf3115ORkeHKMnAgN9xwQ6FOowUCY0yRKngWaoXMLCiVAgRf2MNmofaSvJ18Mpx2WthPXL5+PV+88golS5Zk7/79zHzjDVJSUvhi1y4eeeQRPh4yxL/znj1w6BBL167lqxEj2HfwIE27duWvXbvyn1mzqFW9OlN8hdizfz+ZWVncPXQoE597jhpVq/LRtGkMePVVRg4axMv/938Mu/de96QBUKGCm6gmXJ6gmjV5ZcwYAH755ReWLl1K586dWX7xxYwYMYJ777+fm265haO4ORGmTp1KrVq1mDJliq/Ye/J1lsOxQGCMKR4ys+DwYdAyULp07vv/8ov7uWULnHQSVK6cY5c/depEyZIlAXfxvvXxx1mxfj1SujSZhw+H/djLzz2XMqVLU6Z0aWpWrcqWHTto2agR/V54gYdeeokrzj2X89q0YeHKlSxcvZqL77wTgOxjxzg1UvfOOnWgUiUXvELVrcu3s2Zx9913A9CsWTPq1avH8uXLad++PU899RQZO3Zw7bXXcvrpp9OyZUv69evHQw89xBVXXMF5552X+7nKhQUCY0yRKnAW6s07XL/+KHf4gJvoffXq4F4+y5e7ev+9e4PSMFQoV+635YEjRtCxbVsmDB3K2o0bubBPH8Ip40sOB1CyRAmysrNpUq8e80aPZup33/HwK6/QuV07/tixI2c2bMiskSNz/928qqsIbQaRcr51796ddu3aMWXKFLp06cJbb73FRRddxLx585g6dSoPP/wwnTt3ZtCgQbmXIQrrNWSMKR68i6V3UTx0CNasyZlfZ+NGty3UwYMuIESYvGXPgQPUrlkTcI25uQoIIhu3baN82bL0uOwy+vXowU/LltG0Xj227dvHrJ9/BiAzK4tFq1YBUKl8efYdPOj/rBLRL7Xnn38+77//PgDLly9n/fr1NG3alNWrV9OwYUPuuecerrrqKn7++Wc2btxI+fLl6dGjB/369eOnwCR2BWRPBMaY+MvOdhfuaFMvrlrlqopOOSXoosyuXeH3zyXVw99vvplbH3+cf73/Phelh03CGSwgAP2yciUPvvgiJUQolZLCa/37U7pUKca99Rb3DBrEnm3bXKrrbt04s1Ejel55JX2eftrfWBzy0X95+mnu+9e/ADitYUO++uor+vTpQ8uWLUlJSWHUqFGUKVOGjz76iPfee49SpUpxyimnMGjQIObMmcODDz5IiRIlKFWqFK+99lruv0suLA21MabQ8p2Ges0al7KhRQvwetRs3eqSsNWoAfXquYbgzMzgfcA/J3Cs1arlnj4AUlJyJpYTcWUrUyb3MgX+DqH75iUo5VN+01Bb1ZAxJv68apNwo3n37HEXy3ANq/G6ca1Rw/Xp95x+evD2ChWgbVt/3/6qVYO3V64MDRq4QWBNmwYHstD9igGrGjLGJE5mpru4i/jbCAJHAof69dei+d5TT4VNmyJvr1cvOEiFXshDn34qVw6usqpZMzhpXTht28Y0bUR+2BOBMaZIFKiaecWK3NM4eJ+7enXQ3L+FUq2afzk93VX9gLuzr1DBLQf29Mntgl29enCX12htH54YBYGC/B1i9kQgIiOBK4Ctqtoiwj4XAs8DpYDtqnpBrMpjjImdsmXLsmPHDlJTU5HQC9zOna4KxbvAhjpwAFJTI18Ylywp+glZQnvxeBfPevX8QUEEzjorZ7nCVeeIuOOOHoXmzcE3diGsRo1g+/aClz0KVWXHjh2UzWUkc6hYVg2NAl4GRofbKCJVgFeBS1R1vYjUDLefMab4q1OnDhkZGWwLTLR2+LC74HndOevV82/bssVfBbRvnxtxW7p0/HL+r1jhvxh7gWb3btf9NFJA8vYvX94dE2rrVvc7r1z520T1UYX7jCJQtmxZ6tSpk69jYhYIVHWmiNSPskt3YLyqrvftvzVWZTHGxFapUqVo0KCBu/vftcvd9bZtGzxRO8Do0XDzzdCtm+sVFOidd+DWW+NT4EOHoGVLt5zXqhQvXUSk/VNTYfx4uOSSwpcvzhLZRtAEqCoiX4vIPBG5JdKOItJbROaKyNxthZ1WzhgTO61aQePGbjncQKdbbnFVQeEuphEGggVp2jTytmuuyVMRAX9vn9wadENFGI0MuAbiaNuLsUQGghSgLXA50AUYKCJNwu2oqm+oarqqptfIb1pXY0zhrVrlqnCWLnUX3CNHwu/n9eqJNtrVm/0r1Cef5F6O9u0jb4uUpvmcc3KuE4H334c5c3L/To8qFMHgreIokd1HM3ANxAeAAyIyE2gNLE9gmYwx4TRuDO3aubrvb7+F2bPh/PMj79+2beRtH3wQfv2kSbmXY9myyNtSwlzOateGNm3g++9zbuvePffvSxKJfCKYCJwnIikiUh5oB8Sm9cSYZLd+PezfX7jPmD3bNYZC8Py9CxYEjwPIzbPPgi8/T77NmhV5W7gG2uxseOqp4HXhAkaSi1kgEJEPgVlAUxHJEJHbRaSPiPQBUNUlwH+An4EfgbdUdWGsymNMUqtXDy66KP/HZWb6RwGDP9nbFVe4nxMnuskCEqFSpeC2hnCB4Ngxl6La07w5fPxx7Mt2nIlZIFDVbqp6qqqWUtU6qvq2qo5Q1REB+wxV1eaq2kJVn49VWYxJWjNm+KtF5syBMWNg2jR3AVV1XSZFXI+X//0v5/HnnRfc/z8wh/+PP7pAEA+jRrmf9ev713lB4JVX3M9wd/qhA7sWLYKrrirq0h33bGSxMSeyCy+EDh3877t1gy5d3ICq3r398/Lu3OkagX/9NTjHz+zZwZ8XOLK3XTv4979jVfLg7/nDH9xyYGqH225zP8uXdz9r1IChQ/3bhw51Qc/kyirLjDlReHfIea2rf+stuOsu//vDh91MWn36uN4xb76Z85gDBwpfzvyoUweGD3eNvhMnugZqL8Hb8OHu5803u0R1ffq4nkP167tyxmtMwgnAAoExJ4qTToK6deHPf3ZdNF96Kfdjlgd00tvqG9M5frzLnPnQQ7EpZ34sW+a/4w+t0vHSRJQsCffe61/ftWvOz5k501WDmbAsEBhzoti3z9WB33efe5+XEa7XX59z3ZEjxSMINGwYPn3zl1+6BHT5UQTz+p7IrI3AmOJi6dLwOfjzItxxDzxQsM/as6dgx4VTu3bw+yefhJ4983bsqlXhp3js2BFuv73QRTN+FgiMKQ5+/dU1hHp389HMmOG6RU6Y4Bp5ly4NToHsiTb4qqh17w7NmuVcn5HhqqsA3n0XHn0094Rs//63v/7fxIVVDRlTHHiZLV991aU5fvpp/7adO6FiRTei95NPXN3//fe7i2WnTjB9ekKK/Jt164Iv9reEpA3zpnhs2DD4fTglS+b9icEUGXsiMKa4GTLEv6zq5s5t185d9L0GYC8dQ6KDAPiDAITPw+9d+L1cQIHJ5WrWdAPW2rTJuc3EjQUCY4qDSKmNjx1zjbfz5wevX7Uq5kUKK3RU7t13B7+PFgi8ht++fV0A2LzZzUtQrhz86U9FX1aTZ1Y1ZEyiHTzoT9kQaMUKf5fORGjQANasCV5Xvz6cfLK7gIeTlyeC1q1zHh+ujcPEjT0RGJMo+/a5xtRp08JPyt6kCZx7buy+v0cP//KqVcGzg9WvHz5Fc+gFOzStQ2Ag+OEH99N7EoiUJjrc55q4skBgTFE6ciRnWoZrr3UX9VDt2sFppwXn7/GMGJFzXVELHIHcsKGb0H37dpe///PPXf6h0J5HpUsHV2OFBoLA9w0auJ/eTGDR5vG1QJBQFgiMKYxt24JHrN59N/z+98EDniZMcNU8obw5a7t1y7ntr38t0mIC4SdoCZWaCt995w9cTZoEzwpWujSMHOl/H3pxL1fOv1zTNw352LHuVatW5O/1AkHgU4qJGwsExhRGzZrurt7j5csPzNfv6dfP3YV/951rBI630O/Ma06iwP1KlYLLL3d1/O3bB+cqgvB39lWrhk/7EMgbOBZuAJmJOTvrxhRW4IQvXsNoWhpMmRK833PPuZ/nnpuYyVEuvLBgxwUGAu9CX7OmS28dOnL4vPNc1s/A9oa88LqNRqs+MjFjgcCY3Ozf768r/+orV3+/a5e/77vn1VfdKF/PFVdEvuuO1F00li69NPh9Xp8IAucJzq0uX8Q9+VSrlr+yeU8r9kSQEHbWjcnNFVf40ydcdJGrv7/yyuC+/SefDHfeGf+yhWbkbN0a/vlP1xPp2LHghugOHeDvf/e/9/rue717Inn1Vf9ybukhCsoLMN4MaCauRBNxZ1II6enpOnfu3EQXwyQT78557NjiN/CpSxfXw8eTlZWzesUrf+h8Bfn5v+8dk5kZm2qtjAzX1jJ1as4nF1MkRGSeqqaH22YDyozJyIBx41xO+8DqkjlzgqdvLA5B4NxzXc4hcHfRodU74erY27RxXVU9o0blv7F6yBDo3z92dfh16iSmuswA9kRgkpGqm8GqYkX3/rzz3MV1+XI4/XQ4etTl8knkhb9WLdi4Mef6Xbtcw+qUKXDjja73zksv+adoPM7+P5v4ifZEYG0E5sSydq2bbH3xYvf+mWfg4Yfd8qxZbpDTk0+6Sc1F4B//8N9he+MBBg5M/N3/zz8Hv3/tNXeRr1LF9fW/5Rb3RHDaafDss8FdWI3JJwsE5vhz6BBs2JBz/aZN8MEHLnfPO++4CVb693fVGsuWwSOPuEDx2GP+YwYO9C8fPgxffOEPDPmVW1/5QNEmfVf1z8vrSQ97I+c3bx7Yk7IpIGsjMMefK65w0xUGVoPs2xc8crVMGX+Of4BevfyDvSI5eDBv0zuG07evS6Uwblz47X/7m7/3Tbt2Lud+Zib07u16+GRlwWWX+fcvUcKN4O3QwfVIOumk6N9fo4Z7GVMA9kRg4mvs2MiZK3OzerWr2vnyS/f+ySf92/buDd63VCk3f68ntyAA/pQPBVG+fM6G1K+/9i8/+qj7WauWv7vmHXe4KqyLLw7fU6ZXL5fiIbcgYEwhWSAwRWvGDFdFE86uXW6ydC/l8p49ri4+8M7dM2dOzpms+vd3VTueQYPcU8GxYzkbSV94Aa6+On9lv//+/O0fKFwguOACNxjtu+/g1FPdQLQZM/zbRdz0lMYkWMwCgYiMFJGtIrIwl/3OFpFsEclHBaspti68ENq2Db/Nu7B7Oe5ff91VpQwd6oLCX/4Cn37qnhp+9zsYPNjdMb//vhuxO3Zszs8sUcJdgEMzeOY3xUFhVa4cflRshQr+ZG9/+Qs0bhzfchmTB7FsIxgFvAyMjrSDiJQEngE+j7SPOQ5FeiLo0iX4vXcHnZ3t0hgsWQJvvOHf/tRT7pUX4bJ7xkpaWs4Zwxo2DE401717/MpjTCHF7IlAVWcCO3PZ7W7gYyCB0zCZIpNbH/bAwVngDwQzZxaufh6CG1qLwgcfuKeMwLTIN9zgfrZokXP/Ro3coChwvZPefTf/37lxY2JnJDNJK2FtBCJSG/gjkOsMHCLSW0Tmisjcbdu2xb5wpmAyM8Ov/9vfwic481IVhJsJK9G6dXM9j+rX96/zJlgRCZ44pnFj9zrnHBfUHn+8YMnTTj3Vev6YhIjpyGIRqQ9MVtUct1AiMhZ4TlV/EJFRvv0i9L3zs5HFxdjBg65OHNzd/65d0Lw5nHJK8H6pqfGvw88v7//FoUOuIbhqVZeCYvBg//aC5OwxJkGK68jidGCMiKwFugKvisg1CSyPyYusLNe/3csfHyjwiaBNG5epM3D2Lk8ig0Bo3X4g7wll8mT/unLlXEDbsMHf9hGag9+Y41zCAoGqNlDV+qpaHxgH/E1VP0lUeUwevfgi3H47vP128PqsrPBVQ4l8ektNdXX8a9a4lA2bN7s0zeH07QudO7vl0IRsVaq4Jx0vqHl5fYw5QcSy++iHwCygqYhkiMjtItJHRPrE6jtNHOz0tf9v3gwffwzffOMGa5UqFb5+OxZz0A4Z4l9etSr8Ps2aufEJ777r6vlbtnQjdAEeeCDn/s895+roIXJOfO9JJr+TrhhTzMWs+6iqhpmRO+K+PWNVDlMI+/a5BtNSpfz14d7EJIH5egInN4+HwIFbDRu6p5HAHPnr17u7+EiGDXNJ3A4ezLm+WjW45prwxz37rJuj99xz3fvZs3MOejPmOGQji41TqhTcc49b/vRTN9CrcmUXCGrWhCNH3LZwk5J40zjGixeUvCRvJUsGV1WddprLLppX113nflap4i72kaZjbNPGjRL2GsR/9zv/YDFjjmMWCJLdd9+5C2tWlstrP22am/4wMA3z9u1QtiysW1d0UxVGGmGbnQ0TJ0Y/1uulU6+ef91ttxWsHJs2wZgxBTvWmBOEBYIT1c6dLp1ydnbORtzVq92sXJCz0Td09G+gjh1hwICiKd9f/xp+fYkSLhBFu6Mviu6aN93kflarFpupF405jlggOFGlprqZt9LTg6s6Bgxwo2BPOw3OPDN/k4WvWVN0deKh9fOhAgdkVa/ur5cvKq++6hp/I1UDGZNELBCciAJntwrtN//Pf/qXFy+ObbVI+fKRt113net5tG6dv0zlyvm3BwaCrCzXO2npUtdDyRvhG5rcbvfu8OMWwklJsd4/xvjYM3Fxc+SIe1WunLf9jx1zdfwi8OuvLgd/uP78X3zhMoPGyrBh0K9f8LqqVd3gMy+zaKDA9MsPPwx9+gT3Bgpc9gavBfZOWrbM5eoPZHn7jSkQeyIobjp0yPsF7dAhd8H0Jmhp1MjV8Xv5/gPddpsbNBUrd9/t8vME6tnTJWrr3dtNFn/llZGPr1o1OPiFCwSBQoOAMabALBAUN/Pm5X1frxrk1Vdd1kqvi2c4Gza4XkEFdfnl4dd7KZlLl87Zc2fQIP9y+fL5y8UfWDUUOtLXGFOkLBAkwooVrudONKruLtrLcb98uevaed11/jtkrwqoRAn45ZeiLePmzW7AlKdq1fBZQrt396dt+MMfgieICe1q+vTTcO218Oc/5/79gYEg2pOEMabQrI0gEbxqjWjdIP/+d3jzTVe/np0dXD+ekuKmVdy/370vUSLnnL2FVaGCGzC1Zo2bJ/i++3I2zr7+es6Lepky/uXQ1NNlyri0FHnhVQ0NGwZ33pmvohtj8scCQSJNnBh5Xt1hw/zL4XrCDB/uX/71V3enXZS8Hj/160cOWBdcEH6egYoV/UGqoLwngoYN3WA2Y0zMWNVQIgXmtJk9O/xFFWD8+KL93kmTom/Pyoo8scqgQW4MwubNkXMMLVniunkWhletZIO9jIk5CwRFYds2N3hr0CDX190zbZob2BVu0nXP99/D6ae71M6R5KVOPS9atXJVPNHq3M86K7jHTqjHH3dJ3bxMnuHUqQO//32Biwn4B3pZIDAm5ux/WVF46y2XzuHbb11Xzm+/hZUrXS+aY8fg+uvhq6/CXxw7dIhfOadO9U+qMnly+G6m06bFrzzReAHAAoExMZecTwTZ2f68+p5ly2DLlvx9jpcmIbCnDLh0CD17Bnd73LjRPR1Eu5OOpRtvDJ4y8vLLXd3/sWPB7RSpqfEvWzhe6ouqVRNbDmOSQHIGgv793QVv926YMcPl32nWzA3I8pQp40a7hrN7t6vSqVABBg7MGVTCuekmFziKunePJ1oyuMWL4cMPw1f5iMAnn8Arr8Do0bEpW0Hceqv7WatWYsthTBKI6eT1sVAkk9fXr+9y3Kxe7XqlBLrmGpgwwd9wu3ix60LZsaPLhbNwoT/XTSLt2+cu7F7vHlXXFnH99cH7ffNN0SdsiwdVl/463Kxnxph8izZ5fXJWwHo9YsKNWP3kEzcK19O8uX95zZrITwmxdOaZru3h3HPdKN6lS10XTXAzZnlVU127wsyZ7snm/fdd7p9IPZGKOxELAsbESXI+ETRu7Oa6nTkTzj+/aApW1GrXduMDLr3UNfIaY0whRHsiSM42Au+JoDgEgfbtXdtB6ICwxx5z9fpvvZWYchljkkbyBYJRo/z5e+LlmWfCr//pJ9foXK6cS72g6k/cJuJ6+lhjqTEmxpIrECxYAL16uVGx8fT3v7sEcrt2wZdf+teHm47Rq6o7Xuv2jTHHneQJBKtWuZTJRW3r1uCLeyTly0OVKq73Uc2abl2FCjn388YZVKlSVCU0xpio8hQIROReEaksztsi8pOIdI514YrU5Mmx+dyqVd3F3btwf/UV/Pe//u3h+vcPHOh+hhu8NXiwaxco6iRyxhgTQZ56DYnIAlVtLSJdgDuBgcC/VfWsWBcwVIF7Dc2Z49IqF5ZqcLWNd/6OHnU5eLzJV7Zvd5OuG2NMMVAUvYa8K99luACwIGBdpC8dKSJbRWRhhO03icjPvtf3ItI6j2UpmLPPjtxLaMgQ//INNwRvu/POnEnj3n7bJVYLvNCXLh08A5cFAWPMcSKvTwT/BmoDDYDWQEnga1VtG+WY84H9wGhVbRFm+znAElXdJSKXAoNVtV1uZSn0OILrrgtO65ye7p4Wxo1zo3DPPhtuvtlt++or/4TvIm5O3T17Cv7dxhiTIEXxRHA70B84W1UPAqWAXtEOUNWZQMQkPKr6varu8r39AaiTx7IUjpeSwTNlivvZtSu88ELwROmBE6JMnVr000EaY0wxkNcUE+2B+ap6QER6AGcBLxRhOW4HPou0UUR6A70B6tatW4Rfi78HjydwQpbAQHDppUX7vcYYU0zk9YngNeCgrx7/78A6oEhSVYpIR1wgeCjSPqr6hqqmq2p6jcLmn3nuOf/y/Pk5t7dq5V/2JkcxxpgTWF4DQZa6xoSrgRdU9QUgzGio/BGRVsBbwNWquqOwn5cngU8ArcO0T7du7W8jsFz4xpgkkNeqoX0i8jBwM3CeiJTEtRMUmIjUBcYDN6vq8sJ8Vr4991z4zKOed96B55+HatXiViRjjEmUvAaCG4DuwG2qutl3ER8a7QAR+RC4EKguIhnAY/iCh6qOAAYBqcCr4vrlZ0Vq0S5yfftG3y5iQcAYkzTynIZaRE4Gzva9/VFVt8asVFEUSRpqY4xJMoXuPioi1wM/An8Crgdmi0jXoiuiMcaYRMlr1dAA3BiCrQAiUgP4AhgXq4IZY4yJj7z2GioRUhW0Ix/HGmOMKcby+kTwHxH5HPjQ9/4GwOZPNMaYE0CeAoGqPigi1wEdcMnm3lDVCTEtmTHGmLjI6xMBqvox8HEMy2KMMSYBogYCEdkHhOtfKoCqauWYlMoYY0zcRA0EqlroNBLGGGOKN+v5Y4wxSc4CgTHGJDkLBMYYk+QsEBhjTJKzQGCMMUnOAoExxiQ5CwTGGJPkLBAYY0ySs0BgjDFJzgKBMcYkOQsExhiT5CwQGGNMkrNAYIwxSc4CgTHGJDkLBMYYk+QsEBhjTJKLWSAQkZEislVEFkbYLiLyooisFJGfReSsWJXFGGNMZLF8IhgFXBJl+6XA6b5Xb+C1GJbFGGNMBDELBKo6E9gZZZergdHq/ABUEZFTY1UeY4wx4SWyjaA2sCHgfYZvXQ4i0ltE5orI3G3btsWlcMYYkywSGQgkzDoNt6OqvqGq6aqaXqNGjRgXyxhjkksiA0EGcFrA+zrAxgSVxRhjklYiA8Ek4BZf76HfA3tUdVMCy2OMMUkpJVYfLCIfAhcC1UUkA3gMKAWgqiOAqcBlwErgINArVmUxxhgTWcwCgap2y2W7AnfG6vuNMcbkjY0sNsaYJGeBwBhjkpwFAmOMSXIWCIwxJslZIDDGmCRngcAYY5KcBQJjjElyFgiMMSbJWSAwxpgkZ4HAGGOSnAUCY4xJchYIjDEmyVkgMMaYJGeBwBhjkpwFAmOMSXIWCIwxJslZIDDGmCRngcAYY5KcBQJjjElyFgiMMSbJWSAwxpgkZ4HAGGOSnAUCY4xJchYIjDEmyVkgMMaYJBfTQCAil4jIMhFZKSL9w2w/SUQ+FZEFIrJIRHrFsjzGGGNyilkgEJGSwCvApUBzoJuINA/Z7U5gsaq2Bi4EnhOR0rEqkzHGmJxi+UTwO2Clqq5W1aPAGODqkH0UqCQiAlQEdgJZMSyTMcaYELEMBLWBDQHvM3zrAr0MnAFsBH4B7lXVY6EfJCK9RWSuiMzdtm1brMprjDFJKZaBQMKs05D3XYD5QC0gDXhZRCrnOEj1DVVNV9X0GjVqFHU5jTEmqcUyEGQApwW8r4O78w/UCxivzkpgDdAshmUyxhgTIpaBYA5wuog08DUA3whMCtlnPdAJQEROBpoCq2NYJmOMMSFSYvXBqpolIncBnwMlgZGqukhE+vi2jwCeBEaJyC+4qqSHVHV7rMpkjDEmp5gFAgBVnQpMDVk3ImB5I9A5lmUwxhgTnY0sNsaYJGeBwBhjkpwFAmOMSXIWCIwxJslZIDDGmCRngcAYY5KcBQJjjElyFgiMMSbJWSAwxpgkZ4HAGGOSXFIHgquugi++yLk+IwMefBCOHHHvf/0Vvv02vmUzxph4iWmuoeJq5Uo4/XS3/OmnsGULjBoFNWpAr15w7rmwbh1UqAB33w116rh9d+yAatUSVmxjjImJpHoiGDAARPxBwHPyyfDQQ3DbbdC6tQsCAI8/Dhde6N8vNRVGj4ZVq+Cnn6BdO1i/PvizXn3Vf3xBbNkCI0fCsRzztBljTGyIauikYcVbenq6zp07N9/HvfQS3HNPDAoE3HsvvPBC8LrzzoMWLaB0abjsMrf8+OPwxhsueNxxh3vyGDgw+Djxzev28MPw1FMuIJQsWbBybd3qjj/llOj7paW516hRBfseY0zxJyLzVDU97EZVPa5ebdu21YJYv14VVJs1U73+etUfflDdvVv17rvdelCtWdP9HDVKdckS//pYvtLTVe+/3y0/8ED4fYYMcb/Dvn2qLVuqfvqp6vz5qv/5j+rw4aoNGqi++aZqVpbb79gx1YkT/cfnxtvv9tsLdGqNMccBYK5GuK4mzRNBQR04AIcOweHD8OWX8O67sGgRjBsHO3fClVf69/3jH127wnvvufctWsDChXErakRPP+2quKpUgaNHXbk+/9w1lmdlBe97xRWQkgL9+7uqL4BZs6BiRTj7bJg2DVavdsfu3u2eNsqXh+++gzJlID3dbV+9Gv7wh+DP3rYNOnWC//s/aJbLhKSHDsHMmdClS+6/3549LpRVqZK382FMMor2RGCBIMZU4eOPoVQpWLHCtTNcdZWrMpowwQWKrVvhnXf8xwwdCk2bugumF1SKswYNYM0at1yxIuzf75bnzIGzzoL//Q/mzoXhw2HZMrct8J/dN9+4YNW3L1x8MTzwgGuY96qqzj8fZsxw1VwPPAD9+rmAdPnlULOmvzpt/Hj3PQ89BJUr563skyZB2bLQOcr0SJmZLthUr57nU2JMsWNVQ8exrVtVFy9WHTtWdfNm//pFi1Q//lj1/fdVf/lF9ZNPVPv0cfuPGaM6ZYqrDmvfXrVEiejVUxUrqj7yiGrjxvGpDgt8de5c8H2bN1etUCH8vq+/7s7TN9+48zJjhqs6e+EF1csuU/3uO9WOHf3779wZ+W/QpYvbx6t68+zfr7p3b+H/xllZOT/bmKKGVQ2Z/DhyBKZPd3fyw4e7u3OA3r3dHfjVV7tqohPNc8+56qWdO2HiRPdktn17cPXfgAFujEnFiq4KDdy+6elQq5Z/v5kzoVUr2LvX9TTLzoZHHnFPhp6sLFe91qKFq1aL1tssMxP+8x9Xdec9ARmTH1Y1ZGJKNfzFaf58d3ErXx4aNYLJk6FrV9fGMnQo1K/vxnFs3gyffQbPP+/aLjyffuraIEaOhNdei9MvUwh//rOrkvr5Z/jvfyPv99lnMG8ePPpo8PpzzoHf/c6dh127oEQJd/6aNnWBAtznrl7tAtQjj7gg/cEHcNdd7m+QkQHXXefOdY0a8PLL0LgxXHJJwX+vzZvhxhvhww/h1FML/jkmsaxqyJwQ9u9Xzc72V+McPKg6aZLqtm3u/ahRql98obpuneq336r27av6+OOqtWq56q9wVUiVKql+/73qY49FrmbyXiVLxr/qLNpr4ED/cq9eqsOGRd73s89U58xRveMOd9xjj6k2aqR6552q996rumGDO59ZWa4qcv16/3l/8kn3Gf375/9vNmaM6rJl/vdHj7pyHj6cc9+//MVV4am6cgwapLpjR/6/04RHlKqhhF/Y8/uyQGBi7fvvXXDp0cO1KWza5C5gnm3b3Po1a1SffdYFm/vvVz3llMgX4gYNVEePVu3UKfEBJK+vUaNcQHj8cf+6zEzX7bptW9fGMn266muvqZ5xhmpGhmvHAtU//Ul1xAj/cQcPupf3vlw51bp1XZfoa65R7dDBv+3oUdfG470/Hhw+rLpnj+o//6l65EiiSxOeBQJjipkff3QXzcxMN6Zl507X8Lxypep776m2aqX6j3+oTp6s2q+f6kcfqb78sv/i2KNH4gNFvF6bNrlztn27e3K55ho35ubYMbc+O1s1LU315pvdujffdE87oPrf/7p9Zs1ygXrUKPfavj1vf6etW/03AYcOuScc73sDVa7sL+/w4aqpqaovvugCaWAnj0SKFgisjcCYE8iWLS5lCrjL0tatrr0hM9ON45gxwzVkN27suvbedJMbuf7WW3Dffa4B/LPPXFfdO+6ADRvcZ11+uUu8uGdPwn61AqtbN2cqGIB9+1xbC7hxQQ8+6Br2R4503YrLlfM37v/0kztf4Lo1f/65a7t59FE4eDC4u/Jpp/nPm+fLL6Fjx+B1s2e73GWBKW/273fHeuNsirJjgDUWG2OKxOHDruHau0AdOuTGx5Qv74LJRRf5x2Q8+6wLQBdd5C6q33zjAtCqVS7VypEjLlBNmuTG1bRo4VK1DBjg8n4dPAhjxiTudy1qX3zhet098YQLPB5Vd54GDHCdKALt2eOCTGYmLFgAbdsWPDgkLBCIyCXAC0BJ4C1VHRJmnwuB54FSwHZVvSDaZ1ogMCa5rFvngkWXLv5BfdWqwY8/unxcTz8NTZq4JxZV9wS0fLnr5uwNynzoIejZ0z0NdeqU0F8n3/r2hX/9yy1PmhTcnTk/EhIIRKQksBy4GMgA5gDdVHVxwD5VgO+BS1R1vYjUVNWt0T7XAoExpqhlZ7sxH0eOuCBz7Bg88wxcc43bvn27q8KpW9elsV+/3lWXtWrlUqk88IB7WhozBnr0cHfwy5e76qPSpV1X3qIweDA89ljBjo0WCGI5H8HvgJWqutpXiDHA1cDigH26A+NVdT1AbkHAGGNioWRJqFo1eF2kC27jxu516FDObX37+pdr1/Yve03Jhw65docSJdx3ZmfD2rVuYGHTpm4cSJ06bp9hw2DIEDe+JD0dpk51P2Mhlk8EXXF3+nf43t8MtFPVuwL2eR5XJXQmUAl4QVVHh/ms3kBvgLp167ZdV5iE/8YYk4SiPRHEcmKacE0aoVEnBWgLXA50AQaKSJMcB6m+oarpqppeo6iesYwxxgCxrRrKAE4LeF8H2Bhmn+2qegA4ICIzgda4tgVjjDFxEMsngjnA6SLSQERKAzcCk0L2mQicJyIpIlIeaAcsiWGZjDHGhIjZE4GqZonIXcDnuO6jI1V1kYj08W0foapLROQ/wM/AMVwX02IwlYsxxiQPG1BmjDFJIFGNxcYYY44DFgiMMSbJWSAwxpgkd9y1EYjINqCgI8qqA9uLsDhFpbiWC4pv2axc+WPlyp8TsVz1VDXsQKzjLhAUhojMjdRYkkjFtVxQfMtm5cofK1f+JFu5rGrIGGOSnAUCY4xJcskWCN5IdAEiKK7lguJbNitX/li58iepypVUbQTGGGNySrYnAmOMMSEsEBhjTJJLmkAgIpeIyDIRWSki/eP83aeJyFciskREFonIvb71g0XkVxGZ73tdFnDMw76yLhORLjEs21oR+cX3/XN966qJyH9FZIXvZ9WA/WNeLhFpGnBO5ovIXhG5LxHnS0RGishWEVkYsC7f50dE2vrO80oReVGkoFOQRy3XUBFZKiI/i8gE31SwiEh9ETkUcN5GxLlc+f67xalcHwWUaa2IzPetj+f5inRtiO+/MVU94V+47KergIZAaWAB0DyO338qcJZvuRJuvoXmwGCgX5j9m/vKWAZo4Ct7yRiVbS1QPWTds0B/33J/4Jl4lyvkb7cZqJeI8wWcD5wFLCzM+QF+BNrjJmz6DLg0BuXqDKT4lp8JKFf9wP1CPice5cr33y0e5QrZ/hwwKAHnK9K1Ia7/xpLlieC3+ZNV9SjgzZ8cF6q6SVV/8i3vw825UDvKIVcDY1T1iKquAVbifod4uRp4x7f8DnBNAsvVCVilqtFGk8esXKo6E9gZ5vvyfH5E5FSgsqrOUvc/dnTAMUVWLlWdpqpZvrc/4CaDiihe5YoioefL47tzvh74MNpnxKhcka4Ncf03liyBoDawIeB9BtEvxDEjIvWBNsBs36q7fI/yIwMe/+JZXgWmicg8cXNDA5ysqpvA/UMFaiagXJ4bCf4PmujzBfk/P7V9y/EqH8BtuLtCTwMR+Z+IzBCR83zr4lmu/Pzd4n2+zgO2qOqKgHVxP18h14a4/htLlkCQl/mTY18IkYrAx8B9qroXeA1oBKQBm3CPpxDf8nZQ1bOAS4E7ReT8KPvG9TyKm9nuKmCsb1VxOF/RRCpHvM/bACALeN+3ahNQV1XbAH2BD0SkchzLld+/W7z/nt0IvtmI+/kKc22IuGuEMhSqbMkSCPIyf3JMiUgp3B/6fVUdD6CqW1Q1W1WPAW/ir86IW3lVdaPv51Zggq8MW3yPmt7j8NZ4l8vnUuAnVd3iK2PCz5dPfs9PBsHVNDErn4jcClwB3OSrIsBXjbDDtzwPV6/cJF7lKsDfLZ7nKwW4FvgooLxxPV/hrg3E+d9YsgSCvMyfHDO+Osi3gSWq+q+A9acG7PZHwOvRMAm4UUTKiEgD4HRcQ1BRl6uCiFTylnGNjQt933+rb7dbcXNLx61cAYLu1BJ9vgLk6/z4Hu33icjvff8Wbgk4psiIyCXAQ8BVqnowYH0NESnpW27oK9fqOJYrX3+3eJXL5w/AUlX9rVolnucr0rWBeP8bK0yL9/H0Ai7DtcivAgbE+bvPxT2m/QzM970uA94FfvGtnwScGnDMAF9Zl1HInglRytUQ1wNhAbDIOy9AKjAdWOH7WS2e5fJ9T3lgB3BSwLq4ny9cINoEZOLuum4vyPkB0nEXwFXAy/hG9RdxuVbi6o+9f2MjfPte5/v7LgB+Aq6Mc7ny/XeLR7l860cBfUL2jef5inRtiOu/MUsxYYwxSS5ZqoaMMcZEYIHAGGOSnAUCY4xJchYIjDEmyVkgMMaYJGeBwJg4EpELRWRyosthTCALBMYYk+QsEBgThoj0EJEfffnoXxeRkiKyX0SeE5GfRGS6iNTw7ZsmIj+Ifx6Aqr71jUXkCxFZ4Dumke/jK4rIOHFzB7yfr7zxxsSABQJjQojIGcANuIR8aUA2cBNQAZf76CxgBvCY75DRwEOq2go3gtZb/z7wiqq2Bs7BjWwFl2HyPlxu+YZAhxj/SsZElZLoAhhTDHUC2gJzfDfr5XBJv47hT072HjBeRE4CqqjqDN/6d4CxvhxOtVV1AoCqHgbwfd6P6sttI25WrPrAtzH/rYyJwAKBMTkJ8I6qPhy0UmRgyH7R8rNEq+45ErCcjf0/NAlmVUPG5DQd6CoiNeG3+WPr4f6/dPXt0x34VlX3ALsCJi+5GZihLqd8hohc4/uMMiJSPp6/hDF5ZXcixoRQ1cUi8ihu5rYSuIyVdwIHgDNFZB6wB9eOAC5N8AjfhX410Mu3/mbgdRF5wvcZf4rjr2FMnln2UWPySET2q2rFRJfDmKJmVUPGGJPk7InAGGOSnD0RGGNMkrNAYIwxSc4CgTHGJDkLBMYYk+QsEBhjTJL7f1YOwvRuAnPbAAAAAElFTkSuQmCC"/>

## How to stop training in advance


You can utilize earlystopping to void the model training. If there is no improvement for certain amount of epochs, you can stop the model to train.

- from tensorflow.keras.callbacks import **EarlyStopping**

- EarlyStopping(monitor = "", patience = #)



```python
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss', patience = 20)
```


```python
model.fit(X_tn, y_tn, epochs = 2000, batch_size = 500, validation_split = 0.25, callbacks = [early_stop])
```

<pre>
Epoch 1/2000
2/2 [==============================] - 0s 43ms/step - loss: 0.6068 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7114 - val_sparse_categorical_accuracy: 0.5415
Epoch 2/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6071 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7241 - val_sparse_categorical_accuracy: 0.5459
Epoch 3/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6064 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7402 - val_sparse_categorical_accuracy: 0.5502
Epoch 4/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6070 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7112 - val_sparse_categorical_accuracy: 0.5459
Epoch 5/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6076 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7274 - val_sparse_categorical_accuracy: 0.5415
Epoch 6/2000
2/2 [==============================] - 0s 15ms/step - loss: 0.6092 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7212 - val_sparse_categorical_accuracy: 0.5677
Epoch 7/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6090 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.6803 - val_sparse_categorical_accuracy: 0.5721
Epoch 8/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6105 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7258 - val_sparse_categorical_accuracy: 0.5502
Epoch 9/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6097 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.7560 - val_sparse_categorical_accuracy: 0.5546
Epoch 10/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6151 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7112 - val_sparse_categorical_accuracy: 0.5721
Epoch 11/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6089 - sparse_categorical_accuracy: 0.7518 - val_loss: 1.7079 - val_sparse_categorical_accuracy: 0.5502
Epoch 12/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6106 - sparse_categorical_accuracy: 0.7343 - val_loss: 1.7279 - val_sparse_categorical_accuracy: 0.5502
Epoch 13/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6076 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.7202 - val_sparse_categorical_accuracy: 0.5721
Epoch 14/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6096 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7381 - val_sparse_categorical_accuracy: 0.5415
Epoch 15/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6058 - sparse_categorical_accuracy: 0.7401 - val_loss: 1.7605 - val_sparse_categorical_accuracy: 0.5240
Epoch 16/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6110 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7052 - val_sparse_categorical_accuracy: 0.5502
Epoch 17/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6086 - sparse_categorical_accuracy: 0.7416 - val_loss: 1.6952 - val_sparse_categorical_accuracy: 0.5677
Epoch 18/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6056 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7334 - val_sparse_categorical_accuracy: 0.5502
Epoch 19/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6111 - sparse_categorical_accuracy: 0.7372 - val_loss: 1.7417 - val_sparse_categorical_accuracy: 0.5590
Epoch 20/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6128 - sparse_categorical_accuracy: 0.7489 - val_loss: 1.7299 - val_sparse_categorical_accuracy: 0.5721
Epoch 21/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6047 - sparse_categorical_accuracy: 0.7504 - val_loss: 1.7648 - val_sparse_categorical_accuracy: 0.5459
Epoch 22/2000
2/2 [==============================] - 0s 18ms/step - loss: 0.6085 - sparse_categorical_accuracy: 0.7431 - val_loss: 1.7647 - val_sparse_categorical_accuracy: 0.5328
Epoch 23/2000
2/2 [==============================] - 0s 20ms/step - loss: 0.6125 - sparse_categorical_accuracy: 0.7358 - val_loss: 1.7326 - val_sparse_categorical_accuracy: 0.5546
Epoch 24/2000
2/2 [==============================] - 0s 19ms/step - loss: 0.6105 - sparse_categorical_accuracy: 0.7387 - val_loss: 1.7653 - val_sparse_categorical_accuracy: 0.5371
Epoch 25/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6093 - sparse_categorical_accuracy: 0.7445 - val_loss: 1.7697 - val_sparse_categorical_accuracy: 0.5066
Epoch 26/2000
2/2 [==============================] - 0s 16ms/step - loss: 0.6163 - sparse_categorical_accuracy: 0.7460 - val_loss: 1.7135 - val_sparse_categorical_accuracy: 0.5459
Epoch 27/2000
2/2 [==============================] - 0s 17ms/step - loss: 0.6097 - sparse_categorical_accuracy: 0.7474 - val_loss: 1.7222 - val_sparse_categorical_accuracy: 0.5633
</pre>
<pre>
<keras.callbacks.History at 0x7f7b8e599760>
</pre>
Instead of running to 2000, this model stop training after 27 epochs.


## Conclusion


This post contains several features that you can utilize when building a model. 

