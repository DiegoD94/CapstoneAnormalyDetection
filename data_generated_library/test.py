from time_series import StatFunc, ARMA, TimeSeries
import random
import matplotlib.pyplot as plt
import math

"""
figure 1
2 mixed gaussian
"""
gauss0 = StatFunc(random.gauss, {"mu": 0, "sigma": 1})

gauss1 = StatFunc(random.gauss, {"mu": 20, "sigma": 1})

ts = TimeSeries([gauss0, gauss1], [100, 1])

x = []
y = []
c = []
for t in range(1000):
    t, v, color = ts.next()
    x.append(t)
    y.append(v)
    c.append(color)

plt.figure()
ax = plt.gca()
ax.scatter(x, y, c=c)
plt.savefig("example/f1.png")

"""
figure 2
gaussian with increasing std
"""
gauss0 = StatFunc(random.gauss, {"mu": 0, "sigma": lambda x: 1+x/10})

ts = TimeSeries([gauss0], [1])

x = []
y = []
for t in range(1000):
    t, v, color = ts.next()
    x.append(t)
    y.append(v)

plt.figure()
ax = plt.gca()
ax.scatter(x, y)
plt.savefig("example/f2.png")

"""
figure 3
gaussian with global trend
"""
gauss0 = StatFunc(random.gauss, {"mu": 0, "sigma": 1})

ts = TimeSeries([gauss0], [1], trend_func=lambda x: x*x/10000)

x = []
y = []
for t in range(1000):
    t, v, color = ts.next()
    x.append(t)
    y.append(v)

plt.figure()
ax = plt.gca()
ax.scatter(x, y)
plt.savefig("example/f3.png")


"""
figure 4
gaussian with global trend and season
"""
gauss0 = StatFunc(random.gauss, {"mu": 0, "sigma": 1})

ts = TimeSeries([gauss0], [1], trend_func=lambda x: x*x/1000, season=100)

x = []
y = []
for t in range(1000):
    t, v, color = ts.next()
    x.append(t)
    y.append(v)

plt.figure()
ax = plt.gca()
ax.scatter(x, y)
plt.savefig("example/f4.png")

"""
figure 5
ARMA(1, 1)
"""
gauss0 = StatFunc(random.gauss, {"mu": 0, "sigma": 1})
arma = ARMA([0, 1], [1], gauss0)

ts = TimeSeries([arma], [1])

x = []
y = []
for t in range(1000):
    t, v, color = ts.next()
    x.append(t)
    y.append(v)

plt.figure()
ax = plt.gca()
ax.scatter(x, y)
plt.savefig("example/f5.png")

"""
figure 6
AWS
"""

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
