---
layout: single
title: "Visdom Tutorial"
categories: 
  - 'Deep Learning'
tag: ['Python','Pytorch','Visdom']
toc: true
toc_sticky: true
author_profile: false
---



---



## Visdom Tutorial

Gain from [Deep Learning Zero to All](https://www.youtube.com/watch?v=bAPzUjfPAlQ&list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv&index=21)


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
```

### Import Visdom


```python
import visdom
vis = visdom.Visdom()
```

    Setting up a new session...


### Local Host

1. Open Terminal
2. Type python -m visdom.server
3. Copy and Paste Local Host Server

### Text


```python
vis.text("Hello World")
vis.text("Hello World", env = 'main')
```




    'window_3af1f6965fdf48'



env ='main' is not necessary.

### Image

#### Loading a single Image


```python
a = torch.randn(3,200,200)
vis.image(a)
```




    'window_3af1f69664d624'



## Images

#### Loading a multiple Image


```python
b = torch.randn(3,3,28,28)
vis.images(b)
```




    'window_3af1f696664b06'



### Loading CIFAR10 Data


```python
cifar_train = dsets.CIFAR10(root='CIFAR_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
cifar_test = dsets.CIFAR10(root='CIFAR_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
```

    Files already downloaded and verified
    Files already downloaded and verified


#### Visualizing one data from CIFAR10


```python
data = cifar_train.__getitem__(0)
data[0].size()
vis.image(data[0])
```




    'window_3af1f6971b1b9e'



#### Image using DataLoader


```python
cifar_loader = torch.utils.data.DataLoader(dataset = cifar_train, batch_size = 32, shuffle = True)
```


```python
for num, value in enumerate(cifar_loader):
    value = value[0]
    print(value.size())
    vis.images(value)
    break
```

    torch.Size([32, 3, 32, 32])


### Line Plot


```python
y_data = torch.randn(6)
plt = vis.line(Y = y_data)
```


```python
y_data = torch.Tensor([1,2,3,4,5])
plt = vis.line(Y = y_data)
```

#### Line Update


```python
y_append = torch.randn(1)
x_append = torch.Tensor([6])
vis.line(Y= y_append, X = x_append, win = plt, update = 'append') #win = plt, plt 로 선의 이름을 정했기 때문
```




    'window_3af1f6972276a2'



#### Multiple Line in a single plot


```python
num = torch.Tensor(list(range(0,10)))
print(num.shape)
num = num.view(-1,1)
print(num.shape)
num = torch.cat((num,num), axis = 1)
print(num.shape)
```

    torch.Size([10])
    torch.Size([10, 1])
    torch.Size([10, 2])



```python
plt = vis.line(X = num, Y = torch.randn(10,2))
```

To draw two or more line, shape of x and y should be the same.


```python
y = torch.randn(10,3)
x = torch.rand(10,3)
print(y.shape)
print(x.shape)
```

    torch.Size([10, 3])
    torch.Size([10, 3])



```python
plt = vis.line(X = x, Y = y)
```


```python
vis.close(env = 'main')
```




    ''



#### Adding Line Information


```python
num = torch.Tensor(list(range(0,10)))
print(num.shape)
num = num.view(-1,1)
print(num.shape)
num = torch.cat((num,num), axis = 1)
print(num.shape)
```

    torch.Size([10])
    torch.Size([10, 1])
    torch.Size([10, 2])



```python
plt = vis.line(X = num, Y = torch.randn(10,2), opts =  dict(title = 'Practice 1', showlegend = True))
```


```python
plt = vis.line(X = num, Y = torch.randn(10,2), opts =  dict(title = 'Practice 1',legend = ['#1','#2'], showlegend = True))
```

### Pytorch Loss Function Update Function


```python
def loss_tracker(loss_plot, loss_value, num):
    vis.line(X = num,
            Y = loss_value,
            win = loss_plot,
            update = 'append')
```


```python
plt = vis.line(Y = torch.Tensor(1).zero_(), opts = dict(title = 'Loss Tracker'))
for i in range(500):
    loss = torch.randn(1) + i
    loss_tracker(plt, loss, torch.Tensor([i]))
```

### Close VIsdom


```python
vis.close(env = 'main')
```




    ''

