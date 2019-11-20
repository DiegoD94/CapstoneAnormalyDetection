from time_series import StatFunc, ARMA, TimeSeries
import random
from numpy.random import binomial

N = 5  # total number of user
T = 60   # total months of observation
T0 = 12   # before T0 all user are good
f = 0.5  # fraction of customer that will switch to bad state
p = 0.1  # probability of accidental failure to pay while on good state

# create random number generator for Z and Y
Z = []
Y = []
label = []
for i in range(N):
    bad = random.random() < f  # whether the customer will turn bad
    turn = -1
    if bad:
        turn = random.randrange(T0, T)  # turn point
        stat_func_z = StatFunc(lambda x: x, {'x': lambda x, t=turn: 0 if x < t else 1})
        stat_func_y = StatFunc(binomial, {"n": 1, "p": lambda x, t=turn, p=p: p if x < t else 1})
    else:
        stat_func_z = StatFunc(lambda: 0, {})
        stat_func_y = StatFunc(binomial, {"n": 1, "p": p})
    z = stat_func_z
    y = stat_func_y
    Z.append(z)
    Y.append(y)
    label.append(turn)

# a sample matrix Z and matrix Y
m_Z = [[] for _ in range(N)]
m_Y = [[] for _ in range(N)]
for t in range(T):
    Z_t = [z(t) for z in Z]
    Y_t = [y(t) for y in Y]
    for i in range(N):
        m_Z[i].append(Z_t[i])
        m_Y[i].append(Y_t[i])

# a sample window slide algorithm
win_size = 5
thres = 4
wins = [0]*N
m_pred = [[] for _ in range(N)]
for t in range(T):
    for i in range(N):
        wins[i] += m_Y[i][t]
    if t > win_size:
        for i in range(N):
            wins[i] -= m_Y[i][t-win_size]
    for i in range(N):
        m_pred[i].append(int(wins[i] > thres))

# a sample for generated data
for i in range(N):
    print(label[i])
    for j in range(T):
        print(m_Z[i][j], end='')
    print()
    for j in range(T):
        print(m_Y[i][j], end='')
    print()
    for j in range(T):
        print(m_pred[i][j], end='')
    print()


