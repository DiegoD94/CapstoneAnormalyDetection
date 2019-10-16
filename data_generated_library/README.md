# data generation

## Introduction<a name="introduction"></a>
This folder contains code for generating data for anomaly detection. 

## Directory<a name="directory"></a>
- [Introduction](#introduction)
- [Directory](#directory)
- [Examples](#examples)
- [Usage](#usage)

## Examples <a name="examples"></a>
Following are some of the time series pattern the library now supports:

<p align="center">
    <img src="https://github.com/DH-Diego/CapstoneAnormalyDetection/blob/master/data_generated_library/example/f1.png?raw=true" height="300">
</p>
<p align="center">
    Figure: mixture of two gaussian with different mean
</p>

<p align="center">
    <img src="https://github.com/DH-Diego/CapstoneAnormalyDetection/blob/master/data_generated_library/example/f2.png?raw=true" height="300">
</p>
<p align="center">
    Figure: a gaussian whose std is increasing with time
</p>

<p align="center">
    <img src="https://github.com/DH-Diego/CapstoneAnormalyDetection/blob/master/data_generated_library/example/f3.png?raw=true" height="300">
</p>
<p align="center">
    Figure: white noise with quadratic trend
</p>

<p align="center">
    <img src="https://github.com/DH-Diego/CapstoneAnormalyDetection/blob/master/data_generated_library/example/f4.png?raw=true" height="300">
</p>
<p align="center">
    Figure: time series with seasonal property
</p>

<p align="center">
    <img src="https://github.com/DH-Diego/CapstoneAnormalyDetection/blob/master/data_generated_library/example/f5.png?raw=true" height="300">
</p>
<p align="center">
    Figure: ARMA(1, 1)
</p>

## Usage <a name="usage"></a>
Here is a example code that can generate a simulated CPU usage data.

```python
from time_series import StatFunc, ARMA, TimeSeries
import random
import matplotlib.pyplot as plt

t0 = 0
t1 = 20000
t2 = 25000
t3 = 35000
t4 = 40000
t5 = 60000

def trend_func(t):
    if t < t1:
        return 48
    elif t < t2:
        return 48 + (45-48)*(t - t1)/(t2 - t1)
    elif t < t3:
        return 45
    elif t < t4:
        return 45 + (40-45)*(t - t3)/(t4 - t3)
    else:
        return 40

def std_func(t):
    if t < t1:
        return 3
    elif t < t2:
        return 3 + (2.5-3)*(t - t1)/(t2 - t1)
    elif t < t3:
        return 2.5
    elif t < t4:
        return 2.5 + (1-2.5)*(t - t3)/(t4 - t3)
    else:
        return 1

gauss = StatFunc(random.gauss, {"mu": 0, "sigma": std_func})

ts = TimeSeries([gauss], [1], trend_func=trend_func)

x = []
y = []
for t in range(t5):
    t, v, color = ts.next()
    x.append(t)
    y.append(v)

plt.figure()
ax = plt.gca()
ax.plot(x, y)
plt.savefig("example/aws.png")
```
The result graph is:
<p align="center">
    <img src="https://github.com/DH-Diego/CapstoneAnormalyDetection/blob/master/data_generated_library/example/aws.png?raw=true" height="300">
</p>
