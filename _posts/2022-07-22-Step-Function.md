---
layout: single
title:  "Step Function"
categories: 
  - 'Deep Learning'
tag: ['Activation Function']
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


## Step Function

x>0, h(x) = 1  

x<=0, h(x) = 0



```python
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
```


```python
step_function(-2)
```

<pre>
0
</pre>

```python
step_function(5)
```

<pre>
1
</pre>
#### Using Numpy



```python
import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(int)
```


```python
step_function(np.array([-2,2,0,1]))
```

<pre>
array([0, 1, 0, 1])
</pre>
#### Visualizing Step Function



```python
x = np.arange(-5,5,0.1)
y = step_function(x)
```


```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(x,y)
plt.ylim(-0.2,1.2)
plt.show()
```

<img src="data:image/png;base64,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"/>
