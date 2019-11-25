import numpy as np
from time_series import HMM

np.set_printoptions(precision=2)

GOOD = 0
STRUGGLE = 1
BAD = 2

ALL = 0
INTEREST = 1
NOTHING = 2

# standard matrix
# good, struggle, bad
m_hidden = [[0.89, 0.1, 0.01],  # good
            [0.1, 0.7, 0.2],  # struggle
            [0, 0, 1]  # bad
           ]

# pay all, pay interest, pay nothing
m_observe = [[0.99, 0.01, 0],  # good
             [0.5, 0.5, 0],  # struggle
             [0, 0.2, 0.8]  # bad
            ]

def generate_m(dists):
    assert len(dists) == 9
    m = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            m[i][j] = dists[3*i+j]()
    for i in range(3):
        m[i] /= m[i].sum()
    return m

def generate_m_hidden():
    dists = []
    dists.append(lambda: np.random.uniform(0.79, 0.99))
    dists.append(lambda: np.random.uniform(0.05, 0.15))
    dists.append(lambda: np.random.uniform(0.01, 0.02))

    dists.append(lambda: np.random.uniform(0.05, 0.15))
    dists.append(lambda: np.random.uniform(0.65, 0.75))
    dists.append(lambda: np.random.uniform(0.15, 0.25))

    dists.append(lambda: 0)
    dists.append(lambda: 0)
    dists.append(lambda: 1)
    return generate_m(dists)

def generate_m_observe():
    dists = []
    dists.append(lambda: np.random.uniform(0.95, 0.99))
    dists.append(lambda: np.random.uniform(0.01, 0.02))
    dists.append(lambda: 0)

    dists.append(lambda: np.random.uniform(0.45, 0.55))
    dists.append(lambda: np.random.uniform(0.45, 0.55))
    dists.append(lambda: 0)

    dists.append(lambda: 0)
    dists.append(lambda: np.random.uniform(0.15, 0.25))
    dists.append(lambda: np.random.uniform(0.75, 0.85))
    return generate_m(dists)

def generate_raw_data(T, N):
    m_Z = []
    m_Y = []
    for _ in range(N):  #  number of customer
        Z = []
        Y = []
        m_hidden = generate_m_hidden()
        m_observe = generate_m_observe()
        hmm = HMM(GOOD, m_hidden, m_observe)
        for _ in range(T):  #  time stamp
            z, y = hmm.next()
            Z.append(z)
            Y.append(y)
        # print
        m_Z.append(Z)
        m_Y.append(Y)
    return m_Z, m_Y

def generate_win_data(m_Z, m_Y, win_size, future):
    assert(len(m_Z) > 0)
    N = len(m_Z)
    T = len(m_Z[0])

    train_data = []
    train_label = []
    for i in range(int(0.8*N)):
        for t in range(T-win_size):
            if m_Z[i][t+win_size-1] == 2:
                break
            train_data.append(m_Y[i][t:t+win_size])
            label = 0
            for l in range(t+win_size, min(T, t+win_size+future)):
                if m_Z[i][l] == 2:
                    label = 1
                    break
            train_label.append(label)
            if m_Z[i][t+win_size] == 2:
                break

    test_data = []
    test_label = []
    for i in range(int(0.8*N), N):
        for t in range(T-win_size):
            if m_Z[i][t+win_size-1] == 2:
                break
            test_data.append(m_Y[i][t:t+win_size])
            label = 0
            for l in range(t+win_size, min(T, t+win_size+future)):
                if m_Z[i][l] == 2:
                    label = 1
                    break
            test_label.append(label)
            if m_Z[i][t+win_size] == 2:
                break
    return train_data, train_label, test_data, test_label
