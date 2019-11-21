from time_series import HMM
import numpy as np
from sklearn.linear_model import LogisticRegression
from hmmlearn.hmm import MultinomialHMM
from functools import reduce


np.set_printoptions(precision=2)

GOOD = 0
STRUGGLE = 1
BAD = 2

ALL = 0
INTEREST = 1
NOTHING = 2

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

T = 60
N = 10000
m_Z = []
m_Y = []
for _ in range(N):  #  number of customer
    Z = []
    Y = []
    hmm = HMM(GOOD, m_hidden, m_observe)
    for _ in range(T):  #  time stamp
        z, y = hmm.next()
        Z.append(z)
        Y.append(y)
    # print
    """
    for t in range(T):
        print(Z[t], end="")
    print()
    for t in range(T):
        print(Y[t], end="")
    print()
    print()
    """
    m_Z.append(Z)
    m_Y.append(Y)

####################################################
# hmm predict
####################################################
print("--------------------------------------------")
print("HMM:")
def preprocess_data_hmm(Y):
    data = np.concatenate([np.reshape(y, (-1, 1)) for y in Y])
    lengths = [len(y) for y in Y]
    return data, lengths
train_data, train_lengths = preprocess_data_hmm(m_Y[:int(0.8*N)])
test_data, test_lengths = preprocess_data_hmm(m_Y[int(0.8*N):])
test_label = reduce(lambda x, y: x + y, m_Z[int(0.8*N):])

hmm = MultinomialHMM(n_components=3, n_iter=50)
hmm.fit(train_data, test_lengths)
print("hidden state transition matrix:")
print(hmm.transmat_)
print("observation matrix:")
print(hmm.emissionprob_)
# print(m_Z[0])
tmp_pred = hmm.predict(np.reshape(m_Z[0], (-1, 1)))
# print(tmp_pred)
mdict = {}
mdict[tmp_pred[0]] = 0
mdict[tmp_pred[-1]] = 2
mdict[3 - tmp_pred[0] - tmp_pred[-1]] = 1
vfunc = np.vectorize(mdict.get)
# print(mdict)
# print(vfunc(tmp_pred))
predict = vfunc(hmm.predict(test_data, test_lengths))
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(predict)):
    if i > 0 and test_label[i-1] == 2 and test_label[i] == 2:
        continue
    if test_label[i] == 2 and predict[i] == 2:
        TP += 1
    if test_label[i] != 2 and predict[i] == 2:
        FP += 1
    if test_label[i] == 2 and predict[i] != 2:
        FN += 1
    if test_label[i] != 2 and predict[i] != 2:
        TN += 1
print("accuracy: ")
print((TP + TN) / (TP + TN + FP + FN))
print("precision: ")
print(TP / (TP + FP))
print("recall: ")
print((TP) / (TP + FN))

####################################################
# logistic regression
####################################################
print("--------------------------------------------")
print("LR:")
# create windows data
win_size = 5
train_data = []
train_label = []
for i in range(int(0.8*N)):
    for t in range(T-win_size):
        if m_Z[i][t+win_size-1] == 2:
            break
        train_data.append(m_Y[i][t:t+win_size])
        if m_Z[i][t+win_size] == 2:
            train_label.append(1)
            break
        else:
            train_label.append(0)

test_data = []
test_label = []
for i in range(int(0.8*N), N):
    for t in range(T-win_size):
        if m_Z[i][t+win_size-1] == 2:
            break
        test_data.append(m_Y[i][t:t+win_size])
        if m_Z[i][t+win_size] == 2:
            test_label.append(1)
            break
        else:
            test_label.append(0)

lr = LogisticRegression(class_weight="balanced")
lr.fit(train_data, train_label)

test_neg_data = []
test_neg_label = []
for i in range(len(test_data)):
    if test_label[i]:
        test_neg_data.append(test_data[i])
        test_neg_label.append(1)

print("accuracy: ")
print(lr.score(test_data, test_label))
print("recall: ")
print(lr.score(test_neg_data, test_neg_label))

for i in range(5):
    print(test_neg_data[i])
    print(lr.predict(test_neg_data[i:i+1]))