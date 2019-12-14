#!/usr/bin/env python
# coding: utf-8

# In[2]:


from credit_card_data import generate_raw_data, generate_win_data
from sklearn.linear_model import LogisticRegression
from functools import reduce
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools


# In[35]:


T = 60
N = 10000
m_Z, m_Y = generate_raw_data(T, N)
win_size = 10
future = 5
train_data, train_label, test_data, test_label = generate_win_data(m_Z, m_Y, win_size, future)


# In[19]:


def evaluate_model(predictions, probs, train_predictions, train_probs, test_labels, train_labels):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    #for metric in ['recall', 'precision', 'roc']:
        #print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();


# In[6]:


import itertools
from mlxtend.plotting import plot_confusion_matrix


# In[36]:


clf = SVC(gamma='auto', class_weight='balanced', kernel='rbf', max_iter= -1, degree=3, probability=True)
clf.fit(train_data, train_label) 
predict_label = clf.predict(test_data)


# In[37]:


confM = confusion_matrix(test_label, predict_label)
print('The confusion table by SVM is \n {} \n'.format(confM))
print('The precision is {} \n'.format(confM[1][1]/(confM[1][1]+confM[0][1])))
print('The recall is {} \n'.format(confM[1][1]/(confM[1][1]+confM[1][0])))


# In[38]:


# Training predictions (to demonstrate overfitting)
from mlxtend.plotting import plot_confusion_matrix

train_rf_predictions = clf.predict(train_data)
train_rf_probs = clf.predict_proba(train_data)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = clf.predict(test_data)
rf_probs = clf.predict_proba(test_data)[:, 1]

# Confusion matrix

cm = confusion_matrix(test_label, rf_predictions)
fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, cmap=plt.cm.Oranges, figsize = (8,8), colorbar = True)
plt.title('SVM Confusion Matrix')
#plt.figure(figsize = (10, 10))
plt.show()

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs, test_label, train_label)
plt.savefig('roc_auc_curve.png')


# In[42]:


## P-R Curve for SVM Algorithm 

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
precision, recall, thresholds = precision_recall_curve(test_label, rf_probs)
area = auc(recall, precision)

plt.clf()
plt.plot(recall, precision, label='SVM curve', linewidth=2, color = 'blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('P-R Curve')
plt.legend(loc="upper right")
plt.show()


# In[ ]:




