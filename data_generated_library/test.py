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