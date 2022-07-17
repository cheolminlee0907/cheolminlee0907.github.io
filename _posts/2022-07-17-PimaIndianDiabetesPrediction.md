---
layout: single
title:  "Deep Learning model for Diabetes Prediction"
categories: 
  - "Deep Learning"
tag: ['Python', 'Deep Learning', 'Sequential','Dense', 'Adam','Diabetes Prediction']
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


This post describes how to build a deep learning model to predict diabetes outcomes for the Pima Indians. Dataset is from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).


## Loading Packages



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
```


```python
diabetes = pd.read_csv("/Users/cheolmin/Documents/machine_learning/Blog_Post/Pima_Indian/diabetes.csv")
diabetes
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>763</th>
      <td>10</td>
      <td>101</td>
      <td>76</td>
      <td>48</td>
      <td>180</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121</td>
      <td>72</td>
      <td>23</td>
      <td>112</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93</td>
      <td>70</td>
      <td>31</td>
      <td>0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 9 columns</p>
</div>


## Describing Data



```python
diabetes.info()
```

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
</pre>

```python
diabetes.isnull().sum()
```

<pre>
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
</pre>

```python
diabetes.describe()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


There are 8 features for diabetes and 768 samples. Also, there is no null data.


However, there are several 0 values for glucose, blood pressure, skin thickness, insulin, and BMI which does not make sense. Therefore, I will replace these values with the average values for each feature.


### Data Manipulation


#### Glucose



```python
avg_glu = diabetes['Glucose'].mean()
diabetes['Glucose'].replace(0, avg_glu, inplace = True)
```

<pre>
44.0
</pre>
#### Blood Pressure



```python
avg_bp = diabetes['BloodPressure'].mean()
diabetes['BloodPressure'].replace(0, avg_bp, inplace = True)
```

#### Skin Thickness



```python
avg_st = diabetes['SkinThickness'].mean()
diabetes['SkinThickness'].replace(0, avg_st, inplace = True)
```

#### Insulin



```python
avg_ins = diabetes['Insulin'].mean()
diabetes['Insulin'].replace(0, avg_ins, inplace = True)
```

#### BMI



```python
avg_bmi = diabetes["BMI"].mean()
diabetes["BMI"].replace(0, avg_bmi, inplace = True)
```

#### After Replacement



```python
diabetes.describe()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>121.681605</td>
      <td>72.254807</td>
      <td>26.606479</td>
      <td>118.660163</td>
      <td>32.450805</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>30.436016</td>
      <td>12.115932</td>
      <td>9.631241</td>
      <td>93.080358</td>
      <td>6.875374</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>44.000000</td>
      <td>24.000000</td>
      <td>7.000000</td>
      <td>14.000000</td>
      <td>18.200000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.750000</td>
      <td>64.000000</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>27.500000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>79.799479</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


### Data Visualization



```python
colormap = plt.cm.gist_heat
plt.figure(figsize = (50,50))
```

<pre>
<Figure size 3600x3600 with 0 Axes>
</pre>
<pre>
<Figure size 3600x3600 with 0 Axes>
</pre>

```python
sb.heatmap(diabetes.corr(), linewidths = 0.1, vmax = 0.5, cmap = colormap,
          linecolor = 'white', annot = True)
plt.show()
```

<img src="data:image/png;base64,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"/>

If you look at the last row (Outcome), glucose, bmi, and age tends to show highest relationship with diabetes outcome.


#### Glucose Histogram



```python
plt.hist(x = [diabetes.Glucose[diabetes["Outcome"] == 0], diabetes.Glucose[diabetes['Outcome'] == 1]],
         bins = 30, histtype = 'barstacked', label = ['Negative',"Positive"])
plt.legend()
plt.xlabel('Glucose Level')
plt.ylabel('Number of Samples')
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAX4AAAEGCAYAAABiq/5QAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAf/UlEQVR4nO3de7hVZdnv8e8vDgIKyEmjEKE2ubXEJS6MUhExFQ8vooVgtiXCTanlqbeEaqsd3qI3rcRMxbSoSEE8Ua8lSmKWZIAuyVOhSUiSKBaiJoe49x9jAAtYhzEXa8zDmr/Pda1rjvHMcbjngnmvZz7zGfdQRGBmZtXjbaUOwMzMisuJ38ysyjjxm5lVGSd+M7Mq48RvZlZl2pc6gCx69+4dAwYMKHUYZmYVZenSpa9ERJ+d2ysi8Q8YMIAlS5aUOgwzs4oi6a8NtXuox8ysyjjxm5lVGSd+M7MqUxFj/GbWtm3atIlVq1bx1ltvlTqUitSpUyf69etHhw4dMm3vxG9mJbdq1Sq6du3KgAEDkFTqcCpKRLB27VpWrVrFwIEDM+3joR4zK7m33nqLXr16Oem3gCR69epV0Kel3BK/pAMk1dX7eU3SRZJ6SrpP0vL0sUdeMZhZ5XDSb7lCf3e5Jf6I+FNE1EREDXAY8CZwJzAFWBARg4AF6bqZmRVJscb4jwWei4i/SjoVGJG2zwQWApcWKQ4zqwADpvxPqx5vxbSTm91GEpdccglXXXUVAFdeeSWvv/46V1xxRavG8vWvf50vfOEL29Y/+MEP8vDDD7fqOZpTrMQ/HrglXd43IlYDRMRqSfs0tIOkycBkgP79+xclSKtgV3TPuN26fOOwirXHHntwxx13MHXqVHr37p3beXZO/MVO+lCEL3cldQRGA7cVsl9EzIiI2oio7dNnl1ITZmatqn379kyePJnvfOc7uzz38ssv8+EPf5ihQ4cydOhQfve7321rP+644xgyZAif/OQn2X///XnllVcAGDNmDIcddhjvfe97mTFjBgBTpkzhX//6FzU1NZx11lkA7LXXXgCMGzeOe+65Z9s5P/7xj3P77bfz73//m8997nMMHTqUwYMHc8MNN+z2ay3GrJ4TgUcj4qV0/SVJfQHSxzVFiMHMrFnnn38+s2bNYt26HT8ZXnjhhVx88cUsXryY22+/nXPOOQeAL3/5y4wcOZJHH32U0047jZUrV27b5+abb2bp0qUsWbKE6dOns3btWqZNm0bnzp2pq6tj1qxZO5xj/PjxzJ49G4CNGzeyYMECTjrpJG666Sa6d+/O4sWLWbx4MTfeeCPPP//8br3OYgz1nMn2YR6AecAEYFr6eHcRYjAza1a3bt04++yzmT59Op07d97Wfv/99/PUU09tW3/ttddYv349v/3tb7nzzjsBGDVqFD16bJ+kOH369G3PvfDCCyxfvpxevXo1eu4TTzyRCy64gA0bNvCrX/2K4cOH07lzZ+bPn8+yZcuYO3cuAOvWrWP58uWZ5+w3JNfEL6kLcBzwyXrN04A5kiYBK4GxecZgZlaIiy66iCFDhjBx4sRtbVu2bGHRokU7/DGA5OKphixcuJD777+fRYsW0aVLF0aMGNHsPPtOnToxYsQI7r33XmbPns2ZZ5657RzXXHMNJ5xwwm6+su1yHeqJiDcjoldErKvXtjYijo2IQenjq3nGYGZWiJ49e3LGGWdw0003bWs7/vjj+d73vrdtva6uDoAjjzySOXPmADB//nz+8Y9/AEmvvEePHnTp0oVnnnmG3//+99v27dChA5s2bWrw3OPHj+eHP/whDz300LZEf8IJJ3Dddddt2+fPf/4zb7zxxm69RpdsMLOyk2X6ZZ4++9nP7pDop0+fzvnnn8/gwYPZvHkzw4cP5/rrr+fyyy/nzDPPZPbs2Rx99NH07duXrl27MmrUKK6//noGDx7MAQccwLBhw7Yda/LkyQwePJghQ4bsMs5//PHHc/bZZzN69Gg6duwIwDnnnMOKFSsYMmQIEUGfPn246667duv1qbGPKuWktrY2fCMWa5Knc1a0p59+mgMPPLDUYRRsw4YNtGvXjvbt27No0SLOPffcbZ8Giq2h36GkpRFRu/O27vGbmbXQypUrOeOMM9iyZQsdO3bkxhtvLHVImTjxm5m10KBBg3jsscdKHUbBXJ3TzKzKOPGbmVUZJ34zsyrjxG9mVmX85a6ZlZ+s03MzH6/5abzt2rXj4IMPZvPmzRx44IHMnDmTLl26ZD7Fiy++yAUXXMDcuXOpq6vjxRdf5KSTTgJg3rx5PPXUU0yZUh63H3GP38wMthVPe+KJJ+jYsSPXX399Qfu/4x3v2FZPp66ubodKm6NHjy6bpA9O/GZmuzjqqKN49tlnefXVVxkzZgyDBw9m2LBhLFu2DIAHH3yQmpoaampqOPTQQ1m/fj0rVqzgfe97Hxs3buSyyy5j9uzZ1NTUMHv2bH70ox/x6U9/mnXr1jFgwAC2bNkCwJtvvsl+++3Hpk2beO655xg1ahSHHXYYRx11FM8880xur8+J38ysns2bN/PLX/6Sgw8+mMsvv5xDDz2UZcuW8fWvf52zzz4bSO7Ode2111JXV8dDDz20Q/G2jh078pWvfIVx48ZRV1fHuHHjtj3XvXt3DjnkEB588EEAfv7zn3PCCSfQoUMHJk+ezDXXXMPSpUu58sorOe+883J7jR7jNzODbTdIgaTHP2nSJN7//vdz++23AzBy5EjWrl3LunXrOOKII7jkkks466yzOP300+nXr1/m84wbN47Zs2dzzDHHcOutt3Leeefx+uuv8/DDDzN27PZixRs2bGjV11efE7+ZGdvH+OtrqJaZJKZMmcLJJ5/MPffcw7Bhw7j//vvp1KlTpvOMHj2aqVOn8uqrr7J06VJGjhzJG2+8wd577120Oj8e6jEza8Tw4cO3VdBcuHAhvXv3plu3bjz33HMcfPDBXHrppdTW1u4yHt+1a1fWr1/f4DH32msvDj/8cC688EJOOeUU2rVrR7du3Rg4cCC33ZbcoTYiePzxx3N7Xe7xm1n5KZMqqldccQUTJ05k8ODBdOnShZkzZwLw3e9+lwceeIB27dpx0EEHceKJJ7J69ept+x1zzDFMmzaNmpoapk6dustxx40bx9ixY1m4cOG2tlmzZnHuuefyta99jU2bNjF+/HgOOeSQXF6XyzJb2+CyzBWtUssyl5NCyjJ7qMfMrMo48ZuZVRknfjMrC5Uw7FyuCv3d5Zr4Je0taa6kZyQ9LekDknpKuk/S8vSxR54xmFn569SpE2vXrnXyb4GIYO3atZmnk0L+s3quBn4VER+R1BHoAnwBWBAR0yRNAaYAl+Ych5mVsX79+rFq1SpefvnlUodSkTp16lTQRWS5JX5J3YDhwMcBImIjsFHSqcCIdLOZwEKc+M2qWocOHRg4cGCpw6gaeQ71vAt4GfihpMck/UDSnsC+EbEaIH3cp6GdJU2WtETSEvcCzMxaT56Jvz0wBLguIg4F3iAZ1skkImZERG1E1Pbp0yevGM3Mqk6eiX8VsCoiHknX55L8IXhJUl+A9HFNjjGYmdlOckv8EfF34AVJB6RNxwJPAfOACWnbBODuvGIwM7Nd5T2r5zPArHRGz1+AiSR/bOZImgSsBMY2sb+ZmbWyXBN/RNQBu9SJIOn9m5lZCfjKXTOzKuPEb2ZWZZz4zcyqjBO/mVmVceI3M6syTvxmZlXGid/MrMo48ZuZVRknfjOzKpN3yQazohjw1s8ybbeitU98RfeM261r7TObtZh7/GZmVcaJ38ysyniox0rDQyRmJeMev5lZlXHiNzOrMk78ZmZVptnEL+lCSd2UuEnSo5KOL0ZwZmbW+rL0+D8REa8BxwN9SG6fOC3XqMzMLDdZEr/Sx5OAH0bE4/XazMyswmRJ/EslzSdJ/PdK6gpsyTcsMzPLS5Z5/JOAGuAvEfGmpF4kwz1mZlaBsvT4AzgIuCBd3xPolOXgklZI+qOkOklL0raeku6TtDx97NGiyM3MrEWyJP7vAx8AzkzX1wPXFnCOYyKiJiJq0/UpwIKIGAQsSNfNzKxIsiT+90fE+cBbABHxD6DjbpzzVGBmujwTGLMbxzIzswJlGePfJKkdyZAPkvqQ/cvdAOZLCuCGiJgB7BsRqwEiYrWkfRraUdJkYDJA//79M57O2pysNX3IVpa59c9rVnmyJP7pwJ3APpL+C/gI8KWMxz8iIl5Mk/t9kp7JGlj6R2IGQG1tbWTdz8zMmtZs4o+IWZKWAseSzN8fExFPZzl4RLyYPq6RdCdwOPCSpL5pb78vsKbl4ZuZWaEaHeNPZ9/0lNSTJDnfQvJ5+qW0rUmS9kzn/CNpT5Irf58A5gET0s0mAHfv3kswM7NCNNXjX0oyRt/QVboBvKuZY+8L3Clp63l+FhG/krQYmCNpErASGFtw1GZm1mKNJv6IGLg7B46IvwCHNNC+lmTYyMzMSiDTHbgknQ4cSdLTfygi7sozKGv7Mt8cvdNHc47ErPpkKcv8feBTwB9Jxug/JamQC7jMzKyMZOnxHw28LyK2zuOfSfJHwMzMKlCWK3f/BNS/gmo/YFk+4ZiZWd6y9Ph7AU9L+kO6PhRYJGkeQESMzis4MzNrfVkS/2W5R2FmZkWT5crdBwEkdau/fUS8mmNcZvlwDR6z5hN/Wiztq8C/SIqziWwXcJmZWRnKMtTzOeC9EfFK3sGYmVn+siT+54A38w7E2oCChlFauYyymWWWJfFPBR6W9AiwYWtjRFzQ+C5mZlausiT+G4Bfk1y0lfUGLGZmVqayJP7NEXFJ7pGYmVlRZLly9wFJkyX13alGv5mZVaAsPf6t5RGn1mvzdE4zswqV5QKu3arLb2Zm5SVrPf73AQcBnba2RcSP8wrKzMzyk+XK3cuBESSJ/x7gROC3gBO/mVkFytLj/wjJLRQfi4iJkvYFfpBvWGZtTCEXt12xLr84zMg2q+dfEbEF2JwWaluDv9g1M6tYWRL/Ekl7AzcCS4FHgT80uUc9ktpJekzSL9L1npLuk7Q8fezRksDNzKxlmk38EXFeRPwzIq4HjgMmRMTEAs5xIfB0vfUpwIKIGAQsSNfNzKxIGk38kvaX1L3e+jHAxcCHJHXMcnBJ/YCT2fE7gVOBmenyTGBMgTGbmdluaOrL3TnAacA6STXAbcA3SL7o/T5wTobjfxf4PNC1Xtu+EbEaICJWS9qnoR3T+wBMBujfv39Dm1iZGfCWK26aVYKmhno6R8SL6fLHgJsj4ipgInB4cweWdAqwJiKWtiSwiJgREbURUdunT5+WHMLMzBrQVI9f9ZZHkpZsiIgtkhreY0dHAKMlnURy4Vc3ST8FXpLUN+3t9yWZJWRmZkXSVI//15LmSLoa6EFSmpk0WW9s7sARMTUi+kXEAGA88OuI+BgwD5iQbjYBuHs34jczswI1lfgvAu4AVgBHRsSmtP3twBd345zTgOMkLSeZJTRtN45lZmYFanSoJyICuLWB9scKPUlELAQWpstrgWMLPYaZmbWOTEXazErFM4XMWl+WK3fNzKwNaeoCrgXp4zeLF46ZmeWtqaGevpKOJpmSeSs7Tu8kIh7NNTIzM8tFU4n/MpI6Ov2Ab+/0XJDM7TczswrT1KyeucBcSf8vIr5axJjMzCxHWe65+1VJo4HhadPCiPhFvmFZWcl8E5Hqm4GTddbRik4fbf2TZ/138Y1dbCfNzuqR9A2S0spPpT8Xpm1mZlaBsszjPxmoSe/ChaSZwGOktXvMzKyyZL2Aa2/g1XS5gJuHmlUmXzhmbVmWxP8N4DFJD5BM6RyOe/tmZhUry5e7t0haCAwlSfyXRsTf8w7MzMzykWmoJ71j1rycYzGzPHj2j+3EtXrMzKqME7+ZWZVpcqhH0tuAZRHxviLFY2XIM1zM2pYme/zp3P3HJfUvUjxmZpazLF/u9gWelPQH4I2tjRExOreozMwsN1kS/5dzj8JKwzV4zKpSlnn8D0raHxgUEfdL6gK0yz80MzPLQ5Yibf8XmAvckDa9E7grx5jMzCxHWYZ6zgcOBx4BiIjlkvZpbidJnYDfAHuk55kbEZdL6gnMBgYAK4AzIuIfLYrerECeoWSWbR7/hojYuHVFUnuSO3A1ux8wMiIOAWqAUZKGkdzVa0FEDAIWpOtmZlYkWRL/g5K+AHSWdBxwG/Dz5naKxOvpaof0J4BTgZlp+0xgTKFBm5lZy2UZ6pkCTAL+CHwSuAf4QZaDS2oHLAX+F3BtRDwiad+09g8RsbqxYSNJk4HJAP37+zKCPHjYw6w6ZZnVsyW9+cojJD32P0VElqEeIuLfQI2kvYE7JWW+AjgiZgAzAGprazOdz8zMmpdlVs/JwHPAdOB7wLOSTizkJBHxT2AhMAp4SVLf9Nh9gTWFhWxmZrsjyxj/VcAxETEiIo4GjgG+09xOkvqkPX0kdQY+BDxDUt55QrrZBODuFsRtZmYtlGWMf01EPFtv/S9k66X3BWam4/xvA+ZExC8kLQLmSJoErATGFhq0mZm1XKOJX9Lp6eKTku4B5pCM8Y8FFjd34IhYBhzaQPta4NgWRWtmZrutqR7/f9Rbfgk4Ol1+GeiRW0Rm1S5zDSWzlmk08UfExGIGYmZmxdHsGL+kgcBnSEosbNveZZnNzCpTli937wJuIrlad0uu0ZiZWe6yJP63ImJ67pGYmVlRZEn8V0u6HJhPUngNgIh4NLeozMwsN1kS/8HA/wFGsn2oJ9J1MzOrMFkS/2nAu+qXZjYzs8qVpWTD48DeOcdhZmZFkqXHvy/wjKTF7DjG7+mcZmYVKEvivzz3KMzMrGiy1ON/sBiBmJlZcWS5cnc92++x25HkFopvRES3PAMza0sKudvZik4fzTESs2w9/q711yWNAQ7PKyAzM8tXllk9O4iIu/AcfjOzipVlqOf0eqtvA2rZPvRjZmYVJsusnvp1+TcDK4BTc4nGzMxyl2WM33X5zczakKZuvXhZE/tFRHw1h3jMLKOsM4U8S8h21lSP/40G2vYEJgG9ACd+M7MK1NStF6/auiypK3AhMBG4Fbiqsf3q7bMf8GPg7SRVPWdExNWSegKzSe7otQI4IyL+0fKXYGZmhWhyOqeknpK+Biwj+SMxJCIujYg1GY69GfhsRBwIDAPOl3QQMAVYEBGDgAXpupmZFUmjiV/St4DFwHrg4Ii4opCeeUSs3nqzlohYDzwNvJNkRtDMdLOZwJiWhW5mZi3RVI//s8A7gC8BL0p6Lf1ZL+m1Qk4iaQBwKPAIsG9ErIbkjwOwT4siNzOzFmlqjL/gq3obImkv4Hbgooh4TVLW/SYDkwH69+/fGqGYWTFd0b2AbdflF4ftolWSe2MkdSBJ+rMi4o60+SVJfdPn+wINfl8QETMiojYiavv06ZNnmGZmVSW3xK+ka38T8HREfLveU/OACenyBODuvGIwM7NdZSnZ0FJHkNyk/Y+S6tK2LwDTgDmSJgErgbE5xlCdMn/Ezl4q2KxNyvpeaWNDUbkl/oj4LdDYgP6xeZ3XzMyalusYv5mZlZ88h3rMrJIUMgvHKpp7/GZmVcaJ38ysyniopw0q5MbeZhWlSmfhtDb3+M3Mqox7/GZtXEXcsMVfLBeVe/xmZlXGid/MrMo48ZuZVRknfjOzKuPEb2ZWZTyrx6zM+DqMKlDim9S4x29mVmWc+M3MqowTv5lZlXHiNzOrMk78ZmZVxrN6KsiAKf9T6hDMrA1wj9/MrMo48ZuZVZncEr+kmyWtkfREvbaeku6TtDx97JHX+c3MrGF59vh/BIzaqW0KsCAiBgEL0nUzMyui3BJ/RPwGeHWn5lOBmenyTGBMXuc3M7OGFXtWz74RsRogIlZL2qexDSVNBiYD9O/fv0jhmVlzKuKOXq2tjd3rt2y/3I2IGRFRGxG1ffr0KXU4ZmZtRrET/0uS+gKkj2uKfH4zs6pX7MQ/D5iQLk8A7i7y+c3Mql6e0zlvARYBB0haJWkSMA04TtJy4Lh03czMiii3L3cj4sxGnjo2r3OamVnzXKvHzNqeQu5w1RbOW6CyndVjZmb5cOI3M6syHuoxM8A3ea8m7vGbmVUZJ34zsyrjxG9mVmWc+M3MqowTv5lZlfGsnjLgm6hbW5THLKE2Veq5hNzjNzOrMk78ZmZVxkM9Zla1qvJuYrjHb2ZWdZz4zcyqjId6ClTIDJwV007OMRIzKzeVMnTkHr+ZWZVx4jczqzIe6smRL8wys4YUcnHbihzO7x6/mVmVceI3M6syJRnqkTQKuBpoB/wgIqblda6swy2egWNW/nyXsNZR9B6/pHbAtcCJwEHAmZIOKnYcZmbVqhRDPYcDz0bEXyJiI3ArcGoJ4jAzq0qlGOp5J/BCvfVVwPt33kjSZGByuvq6pD81crzewCu7G5S+ubtHaFSrxJcTx9Yy5RwblHd8FRmbMh/ilNaKZfu5k9zU0t/b/g01liLxN/Q7jF0aImYAM5o9mLQkImpbI7A8lHN8jq1lyjk2KO/4HFvLtHZspRjqWQXsV2+9H/BiCeIwM6tKpUj8i4FBkgZK6giMB+aVIA4zs6pU9KGeiNgs6dPAvSTTOW+OiCd345DNDgeVWDnH59happxjg/KOz7G1TKvGpohdhtfNzKwN85W7ZmZVxonfzKzKVFzil9RO0mOSfpGu95R0n6Tl6WOPEsa2t6S5kp6R9LSkD5RLfJIulvSkpCck3SKpUyljk3SzpDWSnqjX1mg8kqZKelbSnySdUILYvpX+uy6TdKekvcsltnrP/aekkNS7FLE1FZ+kz6QxPCnpv0sRXyP/rjWSfi+pTtISSYeXKLb9JD2Q5o0nJV2YtufznoiIivoBLgF+BvwiXf9vYEq6PAX4Zgljmwmcky53BPYuh/hILpp7Huicrs8BPl7K2IDhwBDgiXptDcZDUtrjcWAPYCDwHNCuyLEdD7RPl79ZTrGl7fuRTJj4K9C7FLE18bs7Brgf2CNd36dcfnfAfODEdPkkYGGJYusLDEmXuwJ/TmPI5T1RUT1+Sf2Ak4Ef1Gs+lSThkj6OKXJYAEjqRvIf6yaAiNgYEf8sl/hIZnB1ltQe6EJy7UTJYouI3wCv7tTcWDynArdGxIaIeB54lqT0R9Fii4j5EbE5Xf09yfUnZRFb6jvA59nxYsiixtZEfOcC0yJiQ7rNmlLE10hsAXRLl7uz/ZqiYse2OiIeTZfXA0+TdNhyeU9UVOIHvkvyn3tLvbZ9I2I1JL88YJ8SxAXwLuBl4IfpUNQPJO1ZDvFFxN+AK4GVwGpgXUTML4fYdtJYPA2V+XhnkWOr7xPAL9PlkscmaTTwt4h4fKenSh5b6j3AUZIekfSgpKFpeznEdxHwLUkvkLxHpqbtJYtN0gDgUOARcnpPVEzil3QKsCYilpY6lka0J/kYeV1EHAq8QfLRrOTSccFTST4SvgPYU9LHShtVQTKV+SgGSV8ENgOztjY1sFnRYpPUBfgicFlDTzfQVorfW3ugBzAM+BwwR5Ioj/jOBS6OiP2Ai0k/sVOi2CTtBdwOXBQRrzW1aQNtmeOrmMQPHAGMlrSCpKLnSEk/BV6S1BcgfVzT+CFytQpYFRGPpOtzSf4QlEN8HwKej4iXI2ITcAfwwTKJrb7G4imLMh+SJpBU4Tor0oHWMojt3SR/0B9P3xv9gEclvb0MYttqFXBHJP5A8om9d5nEN4Hk/QBwG9uHS4oem6QOJEl/VkRsjSmX90TFJP6ImBoR/SJiAEmZh19HxMdIyj1MSDebANxdovj+Drwg6YC06VjgKcojvpXAMEld0p7WsSRjiOUQW32NxTMPGC9pD0kDgUHAH4oZmJKbB10KjI6IN+s9VdLYIuKPEbFPRAxI3xurSL4k/HupY6vnLmAkgKT3kEx8eKVM4nsRODpdHgksT5eLGlv6vrwJeDoivl3vqXzeE3l9S53nDzCC7bN6egELSP7BFgA9SxhXDbAEWEbyn71HucQHfBl4BngC+AnJbICSxQbcQvJ9wyaSZDWpqXhIhjOeA/5EOgujyLE9SzKmWpf+XF8use30/ArSWT3Fjq2J311H4Kfp/71HgZHl8rsDjgSWksyQeQQ4rESxHUkyVLOs3v+xk/J6T7hkg5lZlamYoR4zM2sdTvxmZlXGid/MrMo48ZuZVRknfjOzKuPEbxVL0r6SfibpL5KWSlok6bT0uRFKK7iWA0mv53jsKyT9Z17Ht7bHid8qUnrBy13AbyLiXRFxGMmFff2a3NHMnPitYo0ENkbE9VsbIuKvEXHNzhvu3CNWck+CAeny2Upq7D8u6Sdp2/6SFqTtCyT1T9vHpvs+Luk3aVs7JbX6F6fbfzLrC5D0bkm/Sj+tPCTpf0vqLmmFpLel23SR9IKkDg1t37JfnVW7ot9s3ayVvJfkKtAWk/Rekqsfj4iIVyT1TJ/6HvDjiJgp6RPAdJJyuJcBJ0TE37T9RiyTSKqdDpW0B/A7SfMjKZXbnBnApyJiuaT3A9+PiJGSHicpI/AA8B/AvRGxSdIu25OWQjArhBO/tQmSriW57H1jRAxtbvvUSGBuRLwCEBFba7V/ADg9Xf4Jyc0wAH4H/EjSHLYX9joeGCzpI+l6d5K6KU0m/rQK4weB25JRKyApowEwGxhHkvjHA99vZnuzgjjxW6V6Evjw1pWIOF/JLQeXNLDtZnYc1uyUPopspWwjPcen0p72yUCdpJr0GJ+JiHsLjP9twD8joqaB5+YB30g/gRwG/BrYs4ntzQriMX6rVL8GOkk6t15bl0a2XUFSIhtJQ0jKGENS9OoMSb3S57YO9TxM0tMGOAv4bfr8uyPikYi4jKS65NbbHZ6bltRF0nuU3ICnSZHUWn9e0th0P0k6JH3udZJKi1eTFCP8d1PbmxXKPX6rSBERksYA35H0eZK7n71BUjp5Z7cDZ0uqAxaT3M+UiHhS0n8BD0r6N/AYyb2ILwBulvS59LgT0+N8S9Igkl7+ApKKjsuAASQ18JVuP6aBGLpIWlVv/dskf1Suk/QloAPJfSa23kVrNkl9+BH19mlqe7PMXJ3TzKzKeKjHzKzKOPGbmVUZJ34zsyrjxG9mVmWc+M3MqowTv5lZlXHiNzOrMv8f9DbCXxdPiN8AAAAASUVORK5CYII="/>

#### BMI Histogram



```python
plt.hist(x = [diabetes.BMI[diabetes["Outcome"] == 0], diabetes.BMI[diabetes['Outcome'] == 1]],
         bins = 30, histtype = 'barstacked', label = ['Negative',"Positive"])
plt.legend()
plt.xlabel('BMI')
plt.ylabel('Number of Samples')
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAX4AAAEGCAYAAABiq/5QAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAdO0lEQVR4nO3de5xXVb3/8ddbLg0oKAJ6phCHfseHaQojjkqpqJiKl4NkB8EsyZ/+pqN2xLSOg7/zELoc4/yOVmKWYmr0OJSDeKMyQ0k8mmSAIqlQpBERBIiKeEOQz++PvRkHmMsenP2dy34/Hw8e373Xd18+C+Uza9Zeey1FBGZmVhx7tHUAZmZWWk78ZmYF48RvZlYwTvxmZgXjxG9mVjBd2zqALPr16xcVFRVtHYaZWYeyaNGilyOi/87lHSLxV1RUsHDhwrYOw8ysQ5H0l4bKc+3qkfRlSc9Lek7STyWVSdpX0sOSlqefffKMwczMdpRb4pf0EeByoCoiDgO6AOOAGmBuRBwEzE33zcysRPJ+uNsV6CGpK9ATWA2cDUxPv58OjM45BjMzqye3Pv6I+Juk64GVwNvAnIiYI2n/iFiTHrNG0n4NnS+pGqgGGDhwYF5hmlk7sGXLFlatWsU777zT1qF0SGVlZQwYMIBu3bplOj63xJ/23Z8NDAJeA+6W9Lms50fENGAaQFVVlScUMuvEVq1aRa9evaioqEBSW4fToUQEGzZsYNWqVQwaNCjTOXl29XwK+HNErI+ILcC9wCeBtZLKAdLPdTnGYGYdwDvvvEPfvn2d9HeDJPr27dui35byTPwrgWGSeir5r3kysBSYDYxPjxkPPJBjDGbWQTjp776W/t3l2cf/lKRZwNPAVuAZkq6bvYCZki4i+eEwJq8YzMxsV7m+wBURk4BJOxVvJmn9m5k1qKLmF616vRVTzmz2GElceeWV3HDDDQBcf/31vPHGG0yePLlVY7nuuuu45ppr6vY/+clP8uSTT7bqPZrTId7ctZxM3jvjcRtb93otuaZZiXzoQx/i3nvvZeLEifTr1y+3++yc+Eud9MGTtJmZAdC1a1eqq6v5zne+s8t369ev5zOf+QxHHXUURx11FL/5zW/qyk855RSGDh3KF7/4RQ488EBefvllAEaPHs2RRx7Jxz/+caZNmwZATU0Nb7/9NpWVlZx//vkA7LXXXgCMHTuWBx98sO6eX/jCF7jnnnt47733+OpXv8pRRx3F4MGDufXWWz9wXZ34zcxSl112GTNmzGDjxh1/I50wYQJf/vKXWbBgAffccw8XX3wxAF/72tcYMWIETz/9NJ/+9KdZuXJl3Tl33HEHixYtYuHChUydOpUNGzYwZcoUevToweLFi5kxY8YO9xg3bhy1tbUAvPvuu8ydO5czzjiD22+/nb333psFCxawYMECbrvtNv785z9/oHq6q8fMLNW7d28uuOACpk6dSo8ePerKH3nkEV544YW6/ddff51NmzbxxBNPcN999wEwcuRI+vR5f+qxqVOn1n3317/+leXLl9O3b99G73366adz+eWXs3nzZh566CGGDx9Ojx49mDNnDkuWLGHWrFkAbNy4keXLl2ces98QJ34zs3quuOIKhg4dyoUXXlhXtm3bNubPn7/DDwNIXp5qyLx583jkkUeYP38+PXv25MQTT2x2nH1ZWRknnngiv/rVr6itreW8886ru8dNN93Eaaed9gFr9j539ZiZ1bPvvvty7rnncvvtt9eVnXrqqXzve9+r21+8eDEAxx13HDNnzgRgzpw5vPrqq0DSKu/Tpw89e/Zk2bJl/Pa3v607t1u3bmzZsqXBe48bN44777yTxx9/vC7Rn3baafzgBz+oO+ePf/wjb7755geqo1v8ZtbuZBl+maerrrpqh0Q/depULrvsMgYPHszWrVsZPnw4t9xyC5MmTeK8886jtraWE044gfLycnr16sXIkSO55ZZbGDx4MAcffDDDhg2ru1Z1dTWDBw9m6NChu/Tzn3rqqVxwwQWMGjWK7t27A3DxxRezYsUKhg4dSkTQv39/7r///g9UPzX2q0p7UlVVFV6IJQcezmntxNKlSznkkEPaOowW27x5M126dKFr167Mnz+fSy65pO63gVJr6O9Q0qKIqNr5WLf4zcx208qVKzn33HPZtm0b3bt357bbbmvrkDJx4jcz200HHXQQzzzzTFuH0WJ+uGtmVjBO/GZmBePEb2ZWMO7jt+a1ZLSOmbV7Tvxm1v60dmMjw/DhLl26cPjhh7N161YOOeQQpk+fTs+ePTPfYvXq1Vx++eXMmjWLxYsXs3r1as444wwAZs+ezQsvvEBNTc1uV6E1uavHzAzqJk977rnn6N69O7fcckuLzv/whz9cN5/O4sWLd5hpc9SoUe0m6YMT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"/>

### Building Model



```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```


```python
diabetes_x = diabetes.iloc[:,:-1]
diabetes_y = diabetes.iloc[:,-1]
```


```python
model = Sequential([
Dense(12, input_dim = 8, activation = 'relu'),
Dense(8, activation = 'relu'),
Dense(6, activation = 'relu'),
Dense(1, activation = 'sigmoid')])
model.summary()
```

<pre>
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_22 (Dense)            (None, 12)                108       

 dense_23 (Dense)            (None, 8)                 104       
                                                                 
 dense_24 (Dense)            (None, 6)                 54        
                                                                 
 dense_25 (Dense)            (None, 1)                 7         
                                                                 
=================================================================
Total params: 273
Trainable params: 273
Non-trainable params: 0
_________________________________________________________________
</pre>

```python
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```


```python
model.fit(diabetes_x, diabetes_y, epochs = 100, batch_size = 5)
```

<pre>
Epoch 1/100
154/154 [==============================] - 0s 811us/step - loss: 0.7803 - accuracy: 0.6133
Epoch 2/100
154/154 [==============================] - 0s 771us/step - loss: 0.6843 - accuracy: 0.6445
Epoch 3/100
154/154 [==============================] - 0s 758us/step - loss: 0.6736 - accuracy: 0.6471
Epoch 4/100
154/154 [==============================] - 0s 861us/step - loss: 0.6716 - accuracy: 0.6510
Epoch 5/100
154/154 [==============================] - 0s 792us/step - loss: 0.6616 - accuracy: 0.6510
Epoch 6/100
154/154 [==============================] - 0s 877us/step - loss: 0.6508 - accuracy: 0.6562
Epoch 7/100
154/154 [==============================] - 0s 778us/step - loss: 0.6496 - accuracy: 0.6510
Epoch 8/100
154/154 [==============================] - 0s 804us/step - loss: 0.6479 - accuracy: 0.6510
Epoch 9/100
154/154 [==============================] - 0s 783us/step - loss: 0.6311 - accuracy: 0.6523
Epoch 10/100
154/154 [==============================] - 0s 802us/step - loss: 0.6236 - accuracy: 0.6510
Epoch 11/100
154/154 [==============================] - 0s 777us/step - loss: 0.6254 - accuracy: 0.6510
Epoch 12/100
154/154 [==============================] - 0s 814us/step - loss: 0.6085 - accuracy: 0.6510
Epoch 13/100
154/154 [==============================] - 0s 786us/step - loss: 0.6095 - accuracy: 0.6510
Epoch 14/100
154/154 [==============================] - 0s 780us/step - loss: 0.6127 - accuracy: 0.6497
Epoch 15/100
154/154 [==============================] - 0s 813us/step - loss: 0.6079 - accuracy: 0.6510
Epoch 16/100
154/154 [==============================] - 0s 817us/step - loss: 0.6011 - accuracy: 0.6510
Epoch 17/100
154/154 [==============================] - 0s 783us/step - loss: 0.5942 - accuracy: 0.6510
Epoch 18/100
154/154 [==============================] - 0s 782us/step - loss: 0.5969 - accuracy: 0.6510
Epoch 19/100
154/154 [==============================] - 0s 808us/step - loss: 0.5918 - accuracy: 0.6693
Epoch 20/100
154/154 [==============================] - 0s 829us/step - loss: 0.5879 - accuracy: 0.6758
Epoch 21/100
154/154 [==============================] - 0s 792us/step - loss: 0.5843 - accuracy: 0.6784
Epoch 22/100
154/154 [==============================] - 0s 798us/step - loss: 0.5841 - accuracy: 0.6771
Epoch 23/100
154/154 [==============================] - 0s 797us/step - loss: 0.5839 - accuracy: 0.6836
Epoch 24/100
154/154 [==============================] - 0s 799us/step - loss: 0.5808 - accuracy: 0.6901
Epoch 25/100
154/154 [==============================] - 0s 802us/step - loss: 0.5854 - accuracy: 0.6758
Epoch 26/100
154/154 [==============================] - 0s 784us/step - loss: 0.5809 - accuracy: 0.6810
Epoch 27/100
154/154 [==============================] - 0s 819us/step - loss: 0.5815 - accuracy: 0.6784
Epoch 28/100
154/154 [==============================] - 0s 830us/step - loss: 0.5788 - accuracy: 0.6771
Epoch 29/100
154/154 [==============================] - 0s 888us/step - loss: 0.5754 - accuracy: 0.6888
Epoch 30/100
154/154 [==============================] - 0s 797us/step - loss: 0.5796 - accuracy: 0.6836
Epoch 31/100
154/154 [==============================] - 0s 780us/step - loss: 0.5715 - accuracy: 0.6862
Epoch 32/100
154/154 [==============================] - 0s 783us/step - loss: 0.5689 - accuracy: 0.6875
Epoch 33/100
154/154 [==============================] - 0s 794us/step - loss: 0.5756 - accuracy: 0.6797
Epoch 34/100
154/154 [==============================] - 0s 779us/step - loss: 0.5721 - accuracy: 0.6823
Epoch 35/100
154/154 [==============================] - 0s 775us/step - loss: 0.5704 - accuracy: 0.6823
Epoch 36/100
154/154 [==============================] - 0s 766us/step - loss: 0.5756 - accuracy: 0.6927
Epoch 37/100
154/154 [==============================] - 0s 778us/step - loss: 0.5678 - accuracy: 0.6862
Epoch 38/100
154/154 [==============================] - 0s 786us/step - loss: 0.5710 - accuracy: 0.6797
Epoch 39/100
154/154 [==============================] - 0s 787us/step - loss: 0.5683 - accuracy: 0.6745
Epoch 40/100
154/154 [==============================] - 0s 765us/step - loss: 0.5718 - accuracy: 0.6797
Epoch 41/100
154/154 [==============================] - 0s 785us/step - loss: 0.5740 - accuracy: 0.6810
Epoch 42/100
154/154 [==============================] - 0s 782us/step - loss: 0.5678 - accuracy: 0.6719
Epoch 43/100
154/154 [==============================] - 0s 772us/step - loss: 0.5635 - accuracy: 0.6836
Epoch 44/100
154/154 [==============================] - 0s 799us/step - loss: 0.5621 - accuracy: 0.7096
Epoch 45/100
154/154 [==============================] - 0s 804us/step - loss: 0.5667 - accuracy: 0.6979
Epoch 46/100
154/154 [==============================] - 0s 847us/step - loss: 0.5644 - accuracy: 0.6992
Epoch 47/100
154/154 [==============================] - 0s 786us/step - loss: 0.5579 - accuracy: 0.7292
Epoch 48/100
154/154 [==============================] - 0s 756us/step - loss: 0.5634 - accuracy: 0.6927
Epoch 49/100
154/154 [==============================] - 0s 772us/step - loss: 0.5665 - accuracy: 0.6940
Epoch 50/100
154/154 [==============================] - 0s 790us/step - loss: 0.5644 - accuracy: 0.7070
Epoch 51/100
154/154 [==============================] - 0s 769us/step - loss: 0.5635 - accuracy: 0.7188
Epoch 52/100
154/154 [==============================] - 0s 775us/step - loss: 0.5569 - accuracy: 0.7083
Epoch 53/100
154/154 [==============================] - 0s 801us/step - loss: 0.5517 - accuracy: 0.7096
Epoch 54/100
154/154 [==============================] - 0s 751us/step - loss: 0.5572 - accuracy: 0.7096
Epoch 55/100
154/154 [==============================] - 0s 776us/step - loss: 0.5558 - accuracy: 0.7174
Epoch 56/100
154/154 [==============================] - 0s 780us/step - loss: 0.5566 - accuracy: 0.7109
Epoch 57/100
154/154 [==============================] - 0s 744us/step - loss: 0.5491 - accuracy: 0.7122
Epoch 58/100
154/154 [==============================] - 0s 751us/step - loss: 0.5557 - accuracy: 0.7096
Epoch 59/100
154/154 [==============================] - 0s 763us/step - loss: 0.5536 - accuracy: 0.7174
Epoch 60/100
154/154 [==============================] - 0s 800us/step - loss: 0.5558 - accuracy: 0.7031
Epoch 61/100
154/154 [==============================] - 0s 741us/step - loss: 0.5529 - accuracy: 0.7057
Epoch 62/100
154/154 [==============================] - 0s 747us/step - loss: 0.5540 - accuracy: 0.7018
Epoch 63/100
154/154 [==============================] - 0s 747us/step - loss: 0.5536 - accuracy: 0.7253
Epoch 64/100
154/154 [==============================] - 0s 760us/step - loss: 0.5498 - accuracy: 0.7135
Epoch 65/100
154/154 [==============================] - 0s 777us/step - loss: 0.5528 - accuracy: 0.7188
Epoch 66/100
154/154 [==============================] - 0s 749us/step - loss: 0.5514 - accuracy: 0.7253
Epoch 67/100
154/154 [==============================] - 0s 748us/step - loss: 0.5462 - accuracy: 0.7188
Epoch 68/100
154/154 [==============================] - 0s 752us/step - loss: 0.5455 - accuracy: 0.7018
Epoch 69/100
154/154 [==============================] - 0s 799us/step - loss: 0.5518 - accuracy: 0.7031
Epoch 70/100
154/154 [==============================] - 0s 750us/step - loss: 0.5494 - accuracy: 0.7096
Epoch 71/100
154/154 [==============================] - 0s 775us/step - loss: 0.5445 - accuracy: 0.7096
Epoch 72/100
154/154 [==============================] - 0s 768us/step - loss: 0.5515 - accuracy: 0.7227
Epoch 73/100
154/154 [==============================] - 0s 770us/step - loss: 0.5431 - accuracy: 0.7201
Epoch 74/100
154/154 [==============================] - 0s 754us/step - loss: 0.5408 - accuracy: 0.7305
Epoch 75/100
154/154 [==============================] - 0s 748us/step - loss: 0.5471 - accuracy: 0.7266
Epoch 76/100
154/154 [==============================] - 0s 753us/step - loss: 0.5419 - accuracy: 0.7240
Epoch 77/100
154/154 [==============================] - 0s 830us/step - loss: 0.5403 - accuracy: 0.7305
Epoch 78/100
154/154 [==============================] - 0s 789us/step - loss: 0.5433 - accuracy: 0.7240
Epoch 79/100
</pre>
<pre>
154/154 [==============================] - 0s 756us/step - loss: 0.5405 - accuracy: 0.7344
Epoch 80/100
154/154 [==============================] - 0s 739us/step - loss: 0.5382 - accuracy: 0.7214
Epoch 81/100
154/154 [==============================] - 0s 793us/step - loss: 0.5381 - accuracy: 0.7266
Epoch 82/100
154/154 [==============================] - 0s 752us/step - loss: 0.5355 - accuracy: 0.7214
Epoch 83/100
154/154 [==============================] - 0s 754us/step - loss: 0.5408 - accuracy: 0.7227
Epoch 84/100
154/154 [==============================] - 0s 747us/step - loss: 0.5342 - accuracy: 0.7344
Epoch 85/100
154/154 [==============================] - 0s 749us/step - loss: 0.5354 - accuracy: 0.7279
Epoch 86/100
154/154 [==============================] - 0s 759us/step - loss: 0.5448 - accuracy: 0.7227
Epoch 87/100
154/154 [==============================] - 0s 756us/step - loss: 0.5378 - accuracy: 0.7201
Epoch 88/100
154/154 [==============================] - 0s 764us/step - loss: 0.5411 - accuracy: 0.6940
Epoch 89/100
154/154 [==============================] - 0s 767us/step - loss: 0.5342 - accuracy: 0.7201
Epoch 90/100
154/154 [==============================] - 0s 765us/step - loss: 0.5366 - accuracy: 0.7305
Epoch 91/100
154/154 [==============================] - 0s 764us/step - loss: 0.5388 - accuracy: 0.7214
Epoch 92/100
154/154 [==============================] - 0s 737us/step - loss: 0.5313 - accuracy: 0.7174
Epoch 93/100
154/154 [==============================] - 0s 734us/step - loss: 0.5370 - accuracy: 0.7201
Epoch 94/100
154/154 [==============================] - 0s 736us/step - loss: 0.5348 - accuracy: 0.7214
Epoch 95/100
154/154 [==============================] - 0s 775us/step - loss: 0.5292 - accuracy: 0.7227
Epoch 96/100
154/154 [==============================] - 0s 760us/step - loss: 0.5306 - accuracy: 0.7409
Epoch 97/100
154/154 [==============================] - 0s 748us/step - loss: 0.5348 - accuracy: 0.7292
Epoch 98/100
154/154 [==============================] - 0s 763us/step - loss: 0.5338 - accuracy: 0.7266
Epoch 99/100
154/154 [==============================] - 0s 747us/step - loss: 0.5281 - accuracy: 0.7253
Epoch 100/100
154/154 [==============================] - 0s 744us/step - loss: 0.5296 - accuracy: 0.7318
</pre>
<pre>
<keras.callbacks.History at 0x7f9918ae5c40>
</pre>
### Conclusion



After 100 epochs, this model shows about 73% accuracy to predict the diabetes outcome for Pima Indians.

