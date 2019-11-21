from time_series import HMM

GOOD = 0
STRUGGLE = 1
BAD = 2

ALL = 0
INTEREST = 1
NOTHING = 2

# good, struggle, bad
m_hidden = [[0.89, 0.1, 0.01],  # good
            [0.1, 0.8, 0.1],  # struggle
            [0, 0, 1]  # bad
           ]

# pay all, pay interest, pay nothing
m_observe = [[0.99, 0.01, 0],  # good
             [0.5, 0.5, 0],  # struggle
             [0, 0.2, 0.8]  # bad
            ]

T = 120
N = 10
for _ in range(N):  #  number of customer
    Z = []
    Y = []
    hmm = HMM(GOOD, m_hidden, m_observe)
    for i in range(T):  #  time stamp
        z, y = hmm.next()
        Z.append(z)
        Y.append(y)
    for i in range(T):
        print(Z[i], end="")
    print()
    for i in range(T):
        print(Y[i], end="")
    print()
    print()