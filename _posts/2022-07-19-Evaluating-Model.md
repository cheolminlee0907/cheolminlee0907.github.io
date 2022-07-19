---
layout: single
title:  "Evaluating Model"
categories: 
  - "Deep Learning"
tag: ['Python', "Train Test Split", "Mine vs Rock",'KFold']
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


This post discusses how to evaluate the models using train_test_split from Tensorflow. Dataset is provided from the [Kaggle](https://www.kaggle.com/datasets/armanakbari/connectionist-bench-sonar-mines-vs-rocks). This data is from Dr. Sejnowski's research to differentiate mines from rocks.


## Load Data and Packages



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
mine = pd.read_csv('/Users/cheolmin/Documents/machine_learning/Blog_Post/MinesorRocks/sonar.all-data.csv')
```


```python
mine
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
      <th>Freq_1</th>
      <th>Freq_2</th>
      <th>Freq_3</th>
      <th>Freq_4</th>
      <th>Freq_5</th>
      <th>Freq_6</th>
      <th>Freq_7</th>
      <th>Freq_8</th>
      <th>Freq_9</th>
      <th>Freq_10</th>
      <th>...</th>
      <th>Freq_52</th>
      <th>Freq_53</th>
      <th>Freq_54</th>
      <th>Freq_55</th>
      <th>Freq_56</th>
      <th>Freq_57</th>
      <th>Freq_58</th>
      <th>Freq_59</th>
      <th>Freq_60</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0200</td>
      <td>0.0371</td>
      <td>0.0428</td>
      <td>0.0207</td>
      <td>0.0954</td>
      <td>0.0986</td>
      <td>0.1539</td>
      <td>0.1601</td>
      <td>0.3109</td>
      <td>0.2111</td>
      <td>...</td>
      <td>0.0027</td>
      <td>0.0065</td>
      <td>0.0159</td>
      <td>0.0072</td>
      <td>0.0167</td>
      <td>0.0180</td>
      <td>0.0084</td>
      <td>0.0090</td>
      <td>0.0032</td>
      <td>R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0453</td>
      <td>0.0523</td>
      <td>0.0843</td>
      <td>0.0689</td>
      <td>0.1183</td>
      <td>0.2583</td>
      <td>0.2156</td>
      <td>0.3481</td>
      <td>0.3337</td>
      <td>0.2872</td>
      <td>...</td>
      <td>0.0084</td>
      <td>0.0089</td>
      <td>0.0048</td>
      <td>0.0094</td>
      <td>0.0191</td>
      <td>0.0140</td>
      <td>0.0049</td>
      <td>0.0052</td>
      <td>0.0044</td>
      <td>R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0262</td>
      <td>0.0582</td>
      <td>0.1099</td>
      <td>0.1083</td>
      <td>0.0974</td>
      <td>0.2280</td>
      <td>0.2431</td>
      <td>0.3771</td>
      <td>0.5598</td>
      <td>0.6194</td>
      <td>...</td>
      <td>0.0232</td>
      <td>0.0166</td>
      <td>0.0095</td>
      <td>0.0180</td>
      <td>0.0244</td>
      <td>0.0316</td>
      <td>0.0164</td>
      <td>0.0095</td>
      <td>0.0078</td>
      <td>R</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0100</td>
      <td>0.0171</td>
      <td>0.0623</td>
      <td>0.0205</td>
      <td>0.0205</td>
      <td>0.0368</td>
      <td>0.1098</td>
      <td>0.1276</td>
      <td>0.0598</td>
      <td>0.1264</td>
      <td>...</td>
      <td>0.0121</td>
      <td>0.0036</td>
      <td>0.0150</td>
      <td>0.0085</td>
      <td>0.0073</td>
      <td>0.0050</td>
      <td>0.0044</td>
      <td>0.0040</td>
      <td>0.0117</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0762</td>
      <td>0.0666</td>
      <td>0.0481</td>
      <td>0.0394</td>
      <td>0.0590</td>
      <td>0.0649</td>
      <td>0.1209</td>
      <td>0.2467</td>
      <td>0.3564</td>
      <td>0.4459</td>
      <td>...</td>
      <td>0.0031</td>
      <td>0.0054</td>
      <td>0.0105</td>
      <td>0.0110</td>
      <td>0.0015</td>
      <td>0.0072</td>
      <td>0.0048</td>
      <td>0.0107</td>
      <td>0.0094</td>
      <td>R</td>
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
      <th>203</th>
      <td>0.0187</td>
      <td>0.0346</td>
      <td>0.0168</td>
      <td>0.0177</td>
      <td>0.0393</td>
      <td>0.1630</td>
      <td>0.2028</td>
      <td>0.1694</td>
      <td>0.2328</td>
      <td>0.2684</td>
      <td>...</td>
      <td>0.0116</td>
      <td>0.0098</td>
      <td>0.0199</td>
      <td>0.0033</td>
      <td>0.0101</td>
      <td>0.0065</td>
      <td>0.0115</td>
      <td>0.0193</td>
      <td>0.0157</td>
      <td>M</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0.0323</td>
      <td>0.0101</td>
      <td>0.0298</td>
      <td>0.0564</td>
      <td>0.0760</td>
      <td>0.0958</td>
      <td>0.0990</td>
      <td>0.1018</td>
      <td>0.1030</td>
      <td>0.2154</td>
      <td>...</td>
      <td>0.0061</td>
      <td>0.0093</td>
      <td>0.0135</td>
      <td>0.0063</td>
      <td>0.0063</td>
      <td>0.0034</td>
      <td>0.0032</td>
      <td>0.0062</td>
      <td>0.0067</td>
      <td>M</td>
    </tr>
    <tr>
      <th>205</th>
      <td>0.0522</td>
      <td>0.0437</td>
      <td>0.0180</td>
      <td>0.0292</td>
      <td>0.0351</td>
      <td>0.1171</td>
      <td>0.1257</td>
      <td>0.1178</td>
      <td>0.1258</td>
      <td>0.2529</td>
      <td>...</td>
      <td>0.0160</td>
      <td>0.0029</td>
      <td>0.0051</td>
      <td>0.0062</td>
      <td>0.0089</td>
      <td>0.0140</td>
      <td>0.0138</td>
      <td>0.0077</td>
      <td>0.0031</td>
      <td>M</td>
    </tr>
    <tr>
      <th>206</th>
      <td>0.0303</td>
      <td>0.0353</td>
      <td>0.0490</td>
      <td>0.0608</td>
      <td>0.0167</td>
      <td>0.1354</td>
      <td>0.1465</td>
      <td>0.1123</td>
      <td>0.1945</td>
      <td>0.2354</td>
      <td>...</td>
      <td>0.0086</td>
      <td>0.0046</td>
      <td>0.0126</td>
      <td>0.0036</td>
      <td>0.0035</td>
      <td>0.0034</td>
      <td>0.0079</td>
      <td>0.0036</td>
      <td>0.0048</td>
      <td>M</td>
    </tr>
    <tr>
      <th>207</th>
      <td>0.0260</td>
      <td>0.0363</td>
      <td>0.0136</td>
      <td>0.0272</td>
      <td>0.0214</td>
      <td>0.0338</td>
      <td>0.0655</td>
      <td>0.1400</td>
      <td>0.1843</td>
      <td>0.2354</td>
      <td>...</td>
      <td>0.0146</td>
      <td>0.0129</td>
      <td>0.0047</td>
      <td>0.0039</td>
      <td>0.0061</td>
      <td>0.0040</td>
      <td>0.0036</td>
      <td>0.0061</td>
      <td>0.0115</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
<p>208 rows × 61 columns</p>
</div>


There are 60 features and last column, Label, contains whether the sample is either roc(R) or mine(M). 



```python
mine['Label'].value_counts()
```

<pre>
M    111
R     97
Name: Label, dtype: int64
</pre>
There are 111 mines and 97 rocks in this data.m



```python
mine.isnull().sum()
```

<pre>
Freq_1     0
Freq_2     0
Freq_3     0
Freq_4     0
Freq_5     0
          ..
Freq_57    0
Freq_58    0
Freq_59    0
Freq_60    0
Label      0
Length: 61, dtype: int64
</pre>

```python
mine.describe()
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
      <th>Freq_1</th>
      <th>Freq_2</th>
      <th>Freq_3</th>
      <th>Freq_4</th>
      <th>Freq_5</th>
      <th>Freq_6</th>
      <th>Freq_7</th>
      <th>Freq_8</th>
      <th>Freq_9</th>
      <th>Freq_10</th>
      <th>...</th>
      <th>Freq_51</th>
      <th>Freq_52</th>
      <th>Freq_53</th>
      <th>Freq_54</th>
      <th>Freq_55</th>
      <th>Freq_56</th>
      <th>Freq_57</th>
      <th>Freq_58</th>
      <th>Freq_59</th>
      <th>Freq_60</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>...</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.029164</td>
      <td>0.038437</td>
      <td>0.043832</td>
      <td>0.053892</td>
      <td>0.075202</td>
      <td>0.104570</td>
      <td>0.121747</td>
      <td>0.134799</td>
      <td>0.178003</td>
      <td>0.208259</td>
      <td>...</td>
      <td>0.016069</td>
      <td>0.013420</td>
      <td>0.010709</td>
      <td>0.010941</td>
      <td>0.009290</td>
      <td>0.008222</td>
      <td>0.007820</td>
      <td>0.007949</td>
      <td>0.007941</td>
      <td>0.006507</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.022991</td>
      <td>0.032960</td>
      <td>0.038428</td>
      <td>0.046528</td>
      <td>0.055552</td>
      <td>0.059105</td>
      <td>0.061788</td>
      <td>0.085152</td>
      <td>0.118387</td>
      <td>0.134416</td>
      <td>...</td>
      <td>0.012008</td>
      <td>0.009634</td>
      <td>0.007060</td>
      <td>0.007301</td>
      <td>0.007088</td>
      <td>0.005736</td>
      <td>0.005785</td>
      <td>0.006470</td>
      <td>0.006181</td>
      <td>0.005031</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001500</td>
      <td>0.000600</td>
      <td>0.001500</td>
      <td>0.005800</td>
      <td>0.006700</td>
      <td>0.010200</td>
      <td>0.003300</td>
      <td>0.005500</td>
      <td>0.007500</td>
      <td>0.011300</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000800</td>
      <td>0.000500</td>
      <td>0.001000</td>
      <td>0.000600</td>
      <td>0.000400</td>
      <td>0.000300</td>
      <td>0.000300</td>
      <td>0.000100</td>
      <td>0.000600</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.013350</td>
      <td>0.016450</td>
      <td>0.018950</td>
      <td>0.024375</td>
      <td>0.038050</td>
      <td>0.067025</td>
      <td>0.080900</td>
      <td>0.080425</td>
      <td>0.097025</td>
      <td>0.111275</td>
      <td>...</td>
      <td>0.008425</td>
      <td>0.007275</td>
      <td>0.005075</td>
      <td>0.005375</td>
      <td>0.004150</td>
      <td>0.004400</td>
      <td>0.003700</td>
      <td>0.003600</td>
      <td>0.003675</td>
      <td>0.003100</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.022800</td>
      <td>0.030800</td>
      <td>0.034300</td>
      <td>0.044050</td>
      <td>0.062500</td>
      <td>0.092150</td>
      <td>0.106950</td>
      <td>0.112100</td>
      <td>0.152250</td>
      <td>0.182400</td>
      <td>...</td>
      <td>0.013900</td>
      <td>0.011400</td>
      <td>0.009550</td>
      <td>0.009300</td>
      <td>0.007500</td>
      <td>0.006850</td>
      <td>0.005950</td>
      <td>0.005800</td>
      <td>0.006400</td>
      <td>0.005300</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.035550</td>
      <td>0.047950</td>
      <td>0.057950</td>
      <td>0.064500</td>
      <td>0.100275</td>
      <td>0.134125</td>
      <td>0.154000</td>
      <td>0.169600</td>
      <td>0.233425</td>
      <td>0.268700</td>
      <td>...</td>
      <td>0.020825</td>
      <td>0.016725</td>
      <td>0.014900</td>
      <td>0.014500</td>
      <td>0.012100</td>
      <td>0.010575</td>
      <td>0.010425</td>
      <td>0.010350</td>
      <td>0.010325</td>
      <td>0.008525</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.137100</td>
      <td>0.233900</td>
      <td>0.305900</td>
      <td>0.426400</td>
      <td>0.401000</td>
      <td>0.382300</td>
      <td>0.372900</td>
      <td>0.459000</td>
      <td>0.682800</td>
      <td>0.710600</td>
      <td>...</td>
      <td>0.100400</td>
      <td>0.070900</td>
      <td>0.039000</td>
      <td>0.035200</td>
      <td>0.044700</td>
      <td>0.039400</td>
      <td>0.035500</td>
      <td>0.044000</td>
      <td>0.036400</td>
      <td>0.043900</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 60 columns</p>
</div>


## Building model without Data Split



```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = mine.iloc[:,:-1]
y = mine.iloc[:,-1]
rep ={
    "M":1,
    'R':0
}
y.replace(rep, inplace = True)
```


```python
model = Sequential([
    Dense(60, input_dim = 60, activation = 'relu'),
    Dense(15, activation = 'relu'),
    Dense(10, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])
```

<pre>
2022-07-19 22:24:42.666030: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
</pre>

```python
model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')
model.fit(X,y,epochs = 50, batch_size = 10)
```

<pre>
Epoch 1/50
21/21 [==============================] - 0s 1ms/step - loss: 0.6963 - accuracy: 0.5048
Epoch 2/50
21/21 [==============================] - 0s 1ms/step - loss: 0.6760 - accuracy: 0.5337
Epoch 3/50
21/21 [==============================] - 0s 1ms/step - loss: 0.6599 - accuracy: 0.6058
Epoch 4/50
21/21 [==============================] - 0s 971us/step - loss: 0.6416 - accuracy: 0.6106
Epoch 5/50
21/21 [==============================] - 0s 986us/step - loss: 0.6209 - accuracy: 0.7260
Epoch 6/50
21/21 [==============================] - 0s 974us/step - loss: 0.5892 - accuracy: 0.7212
Epoch 7/50
21/21 [==============================] - 0s 1ms/step - loss: 0.5664 - accuracy: 0.7500
Epoch 8/50
21/21 [==============================] - 0s 927us/step - loss: 0.5387 - accuracy: 0.7452
Epoch 9/50
21/21 [==============================] - 0s 982us/step - loss: 0.5327 - accuracy: 0.7548
Epoch 10/50
21/21 [==============================] - 0s 958us/step - loss: 0.5037 - accuracy: 0.7837
Epoch 11/50
21/21 [==============================] - 0s 1ms/step - loss: 0.4794 - accuracy: 0.7788
Epoch 12/50
21/21 [==============================] - 0s 1ms/step - loss: 0.4541 - accuracy: 0.8029
Epoch 13/50
21/21 [==============================] - 0s 1ms/step - loss: 0.4575 - accuracy: 0.7981
Epoch 14/50
21/21 [==============================] - 0s 1ms/step - loss: 0.4430 - accuracy: 0.7981
Epoch 15/50
21/21 [==============================] - 0s 964us/step - loss: 0.4247 - accuracy: 0.8221
Epoch 16/50
21/21 [==============================] - 0s 997us/step - loss: 0.4037 - accuracy: 0.8269
Epoch 17/50
21/21 [==============================] - 0s 1ms/step - loss: 0.4008 - accuracy: 0.8029
Epoch 18/50
21/21 [==============================] - 0s 999us/step - loss: 0.3812 - accuracy: 0.8462
Epoch 19/50
21/21 [==============================] - 0s 1ms/step - loss: 0.4132 - accuracy: 0.8173
Epoch 20/50
21/21 [==============================] - 0s 961us/step - loss: 0.3968 - accuracy: 0.7933
Epoch 21/50
21/21 [==============================] - 0s 1ms/step - loss: 0.3852 - accuracy: 0.8077
Epoch 22/50
21/21 [==============================] - 0s 974us/step - loss: 0.3605 - accuracy: 0.8510
Epoch 23/50
21/21 [==============================] - 0s 1ms/step - loss: 0.3563 - accuracy: 0.8413
Epoch 24/50
21/21 [==============================] - 0s 1ms/step - loss: 0.3418 - accuracy: 0.8462
Epoch 25/50
21/21 [==============================] - 0s 959us/step - loss: 0.3279 - accuracy: 0.8798
Epoch 26/50
21/21 [==============================] - 0s 1ms/step - loss: 0.3303 - accuracy: 0.8702
Epoch 27/50
21/21 [==============================] - 0s 968us/step - loss: 0.3281 - accuracy: 0.8365
Epoch 28/50
21/21 [==============================] - 0s 1ms/step - loss: 0.3486 - accuracy: 0.8558
Epoch 29/50
21/21 [==============================] - 0s 976us/step - loss: 0.3167 - accuracy: 0.8750
Epoch 30/50
21/21 [==============================] - 0s 952us/step - loss: 0.3167 - accuracy: 0.8606
Epoch 31/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2954 - accuracy: 0.8798
Epoch 32/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2840 - accuracy: 0.8894
Epoch 33/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2798 - accuracy: 0.8942
Epoch 34/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2677 - accuracy: 0.9087
Epoch 35/50
21/21 [==============================] - 0s 984us/step - loss: 0.2759 - accuracy: 0.9087
Epoch 36/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2662 - accuracy: 0.9038
Epoch 37/50
21/21 [==============================] - 0s 941us/step - loss: 0.2545 - accuracy: 0.9087
Epoch 38/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2522 - accuracy: 0.9038
Epoch 39/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2588 - accuracy: 0.9279
Epoch 40/50
21/21 [==============================] - 0s 935us/step - loss: 0.2399 - accuracy: 0.9183
Epoch 41/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2319 - accuracy: 0.9327
Epoch 42/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2198 - accuracy: 0.9279
Epoch 43/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2164 - accuracy: 0.9231
Epoch 44/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2061 - accuracy: 0.9423
Epoch 45/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2025 - accuracy: 0.9519
Epoch 46/50
21/21 [==============================] - 0s 983us/step - loss: 0.1901 - accuracy: 0.9567
Epoch 47/50
21/21 [==============================] - 0s 1ms/step - loss: 0.1846 - accuracy: 0.9423
Epoch 48/50
21/21 [==============================] - 0s 1ms/step - loss: 0.2078 - accuracy: 0.9231
Epoch 49/50
21/21 [==============================] - 0s 1ms/step - loss: 0.1864 - accuracy: 0.9423
Epoch 50/50
21/21 [==============================] - 0s 995us/step - loss: 0.1772 - accuracy: 0.9471
</pre>
<pre>
<keras.callbacks.History at 0x7f95914dd040>
</pre>
Without splitting data, this model shows 94.7% accuracy. Can we believe this? Lets evaluate after splitting the data into 2 different sets: train and test set.


## Splitting Data with train_test_split



```python
from sklearn.model_selection import train_test_split
X_tn, X_te, y_tn, y_te = train_test_split(X,y, test_size = 0.2)
```


```python
model.fit(X_tn,y_tn, epochs = 50, batch_size = 10)
```

<pre>
Epoch 1/50
17/17 [==============================] - 0s 926us/step - loss: 0.1666 - accuracy: 0.9458
Epoch 2/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1800 - accuracy: 0.9096
Epoch 3/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1400 - accuracy: 0.9639
Epoch 4/50
17/17 [==============================] - 0s 995us/step - loss: 0.1299 - accuracy: 0.9639
Epoch 5/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1356 - accuracy: 0.9458
Epoch 6/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1314 - accuracy: 0.9458
Epoch 7/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1202 - accuracy: 0.9578
Epoch 8/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1144 - accuracy: 0.9759
Epoch 9/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1057 - accuracy: 0.9759
Epoch 10/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1032 - accuracy: 0.9699
Epoch 11/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0911 - accuracy: 0.9819
Epoch 12/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0914 - accuracy: 0.9759
Epoch 13/50
17/17 [==============================] - 0s 1ms/step - loss: 0.1155 - accuracy: 0.9518
Epoch 14/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0834 - accuracy: 0.9880
Epoch 15/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0796 - accuracy: 0.9819
Epoch 16/50
17/17 [==============================] - 0s 938us/step - loss: 0.0713 - accuracy: 0.9880
Epoch 17/50
17/17 [==============================] - 0s 978us/step - loss: 0.0662 - accuracy: 0.9940
Epoch 18/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0808 - accuracy: 0.9880
Epoch 19/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0585 - accuracy: 0.9880
Epoch 20/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0649 - accuracy: 0.9880
Epoch 21/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0572 - accuracy: 0.9880
Epoch 22/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0518 - accuracy: 0.9940
Epoch 23/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0470 - accuracy: 1.0000
Epoch 24/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0545 - accuracy: 0.9819
Epoch 25/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0572 - accuracy: 0.9880
Epoch 26/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0408 - accuracy: 1.0000
Epoch 27/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0350 - accuracy: 1.0000
Epoch 28/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0336 - accuracy: 1.0000
Epoch 29/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0306 - accuracy: 1.0000
Epoch 30/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0418 - accuracy: 0.9940
Epoch 31/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0316 - accuracy: 1.0000
Epoch 32/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0292 - accuracy: 1.0000
Epoch 33/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0256 - accuracy: 1.0000
Epoch 34/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0252 - accuracy: 0.9940
Epoch 35/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0263 - accuracy: 1.0000
Epoch 36/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0224 - accuracy: 1.0000
Epoch 37/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0232 - accuracy: 1.0000
Epoch 38/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0216 - accuracy: 1.0000
Epoch 39/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0288 - accuracy: 0.9940
Epoch 40/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0203 - accuracy: 1.0000
Epoch 41/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0176 - accuracy: 1.0000
Epoch 42/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0158 - accuracy: 1.0000
Epoch 43/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0149 - accuracy: 1.0000
Epoch 44/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0145 - accuracy: 1.0000
Epoch 45/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0133 - accuracy: 1.0000
Epoch 46/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0168 - accuracy: 1.0000
Epoch 47/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0126 - accuracy: 1.0000
Epoch 48/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0118 - accuracy: 1.0000
Epoch 49/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0113 - accuracy: 1.0000
Epoch 50/50
17/17 [==============================] - 0s 1ms/step - loss: 0.0104 - accuracy: 1.0000
</pre>
<pre>
<keras.callbacks.History at 0x7f9550c5b100>
</pre>
This train set model shows 100% accuracy.


## Evaluating model with test set



```python
model.evaluate(X_te, y_te)
```

<pre>
2/2 [==============================] - 0s 2ms/step - loss: 0.4322 - accuracy: 0.8810
</pre>
<pre>
[0.43219447135925293, 0.8809523582458496]
</pre>
Unlike the train set model, the test set model shows about 95% accuracy. It is important to evaluate two different sets to avoid and rule out any possible overfitting when building a machine learning model.




## Saving Model


You can save a model and recall the model anytime when you need to. You can simply use model.save() function to save the model.



model.save("name of model.**hdf5**")



```python
model.save('minemodel.hdf5')
```

## Load Previous Model


You can use load function to call models from previous exercise.



from tensorflow.keras.model import load_model



model = load_model("directory/name of model")



```python
from tensorflow.keras.models import load_model
model2 = load_model("/Users/cheolmin/Documents/machine_learning/EverybodyDeepLearning/minemodel.hdf5")
```


```python
model2.evaluate(X_te, y_te)
```

<pre>
2/2 [==============================] - 0s 2ms/step - loss: 0.4322 - accuracy: 0.8810
</pre>
<pre>
[0.43219447135925293, 0.8809523582458496]
</pre>
You can see the same result with previous example.


## KFold


Instead of train_test_split, you can use KFold to split and evaluate a machine learning model.



from sklearn.model_selection import KFold



k = how many times will you split data



kfold = KFold(n_splits = k, shuffle = True)



```python
from sklearn.model_selection import KFold

k = 5
kfold = KFold(n_splits = k, shuffle = True)
```


```python
X = mine.iloc[:,:-1]
y = mine.iloc[:,-1]
```


```python
acc_score = []


def model_fn():
    model = Sequential([
        Dense(24, input_dim = 60, activation = 'relu'),
        Dense(10, activation = 'relu'),
        Dense(1, activation = 'sigmoid')
    ])
    return model

for train_index, test_index in kfold.split(X):
    X_tn, X_te = X.iloc[train_index,:], X.iloc[test_index,:]
    y_tn, y_te = y.iloc[train_index], y.iloc[test_index]
    model = model_fn()
    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
    history = model.fit(X_tn,y_tn, epochs = 200, batch_size = 10,verbose = 0)
    
    acc_score.append(model.evaluate(X_te,y_te)[1])
```

<pre>
2/2 [==============================] - 0s 2ms/step - loss: 0.5664 - accuracy: 0.7857
2/2 [==============================] - 0s 2ms/step - loss: 0.5302 - accuracy: 0.8333
2/2 [==============================] - 0s 2ms/step - loss: 1.0110 - accuracy: 0.8095
2/2 [==============================] - 0s 2ms/step - loss: 0.7569 - accuracy: 0.8537
2/2 [==============================] - 0s 2ms/step - loss: 0.5707 - accuracy: 0.8293
</pre>

```python
print(f'Accuracy score:{acc_score}')
print(f"Average accuracy score: {np.round(np.mean(acc_score)*100,2)}%")
```

<pre>
Accuracy score:[0.7857142686843872, 0.8333333134651184, 0.8095238208770752, 0.8536585569381714, 0.8292682766914368]
Average accuracy score: 82.23%
</pre>
## Conclusion


Overfitting is one of the most important things to consider when building a machine/deep learning model. There are several ways to check for overfitting. 

