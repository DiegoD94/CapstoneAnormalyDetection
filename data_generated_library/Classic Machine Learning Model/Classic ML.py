#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from credit_card_data import generate_raw_data, generate_win_data
get_ipython().run_line_magic('matplotlib', 'inline')
style.use("ggplot")


# In[17]:


T = 60
N = 10000
m_Z, m_Y = generate_raw_data(T, N)
future = 3
win_size = 5
train_data, train_label, test_data, test_label = generate_win_data(m_Z, m_Y, win_size, future)


# In[18]:


train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)


# In[19]:


from sklearn.linear_model import LogisticRegression

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


# In[20]:


best_model = lr

# Training predictions (to demonstrate overfitting)
train_lr_predictions = best_model.predict(train_data)
train_lr_probs = best_model.predict_proba(train_data)[:, 1]

# Testing predictions (to determine performance)
lr_predictions = best_model.predict(test_data)
lr_probs = best_model.predict_proba(test_data)[:, 1]

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_label, 
                                     [1 for _ in range(len(test_label))])
    baseline['precision'] = precision_score(test_label, 
                                      [1 for _ in range(len(test_label))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_label, predictions)
    results['precision'] = precision_score(test_label, predictions)
    results['roc'] = roc_auc_score(test_label, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_label, train_predictions)
    train_results['precision'] = precision_score(train_label, train_predictions)
    train_results['roc'] = roc_auc_score(train_label, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_label, [1 for _ in range(len(test_label))])
    model_fpr, model_tpr, _ = roc_curve(test_label, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); 
    plt.title('ROC Curves');
    plt.show();

evaluate_model(lr_predictions, lr_probs, train_lr_predictions, train_lr_probs)


# In[21]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

precision, recall, thresholds = precision_recall_curve(test_label, lr_probs)
area = auc(recall, precision)

plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC=%0.2f' % area)
plt.legend(loc="upper right")
plt.show()


# In[22]:


from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Confusion matrix
cm = confusion_matrix(test_label, lr_predictions)
plot_confusion_matrix(cm, classes = ['0', '1'],
                      title = 'Confusion Matrix')


# In[23]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
from pprint import pprint

print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[9]:


# Number of trees in random forest
n_estimators = [20, 40, 60, 80]

# Maximum number of levels in tree
max_depth = [20, 50, 70]
max_depth.append(None)

# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf,
               'class_weight': ['balanced']}
pprint(random_grid)


# In[37]:


from sklearn.model_selection import GridSearchCV

rf_search = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 5)
rf_search.fit(train_data, train_label)


# In[38]:


rf_search.best_params_


# In[24]:


#best_model = rf_search.best_estimator_
best_model = RandomForestClassifier(max_depth = 20, min_samples_leaf = 2,  n_estimators = 20, class_weight = 'balanced')
best_model.fit(train_data, train_label)


# In[25]:


test_neg_data = []
test_neg_label = []
for i in range(len(test_data)):
    if test_label[i]:
        test_neg_data.append(test_data[i])
        test_neg_label.append(1)

print("accuracy: ")
print(best_model.score(test_data, test_label))
print("recall: ")
print(best_model.score(test_neg_data, test_neg_label))


# In[26]:


# Training predictions (to demonstrate overfitting)
train_rf_predictions = best_model.predict(train_data)
train_rf_probs = best_model.predict_proba(train_data)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = best_model.predict(test_data)
rf_probs = best_model.predict_proba(test_data)[:, 1]

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_label, 
                                     [1 for _ in range(len(test_label))])
    baseline['precision'] = precision_score(test_label, 
                                      [1 for _ in range(len(test_label))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_label, predictions)
    results['precision'] = precision_score(test_label, predictions)
    results['roc'] = roc_auc_score(test_label, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_label, train_predictions)
    train_results['precision'] = precision_score(train_label, train_predictions)
    train_results['roc'] = roc_auc_score(train_label, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_label, [1 for _ in range(len(test_label))])
    model_fpr, model_tpr, _ = roc_curve(test_label, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); 
    plt.title('ROC Curves');
    plt.show();

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)


# In[27]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

precision, recall, thresholds = precision_recall_curve(test_label, rf_probs)
area = auc(recall, precision)

plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


# In[32]:


random_forst_precision, random_forst_recall = precision, recall


# In[29]:


from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Confusion matrix
cm = confusion_matrix(test_label, rf_predictions)
plot_confusion_matrix(cm, classes = ['0', '1'],
                      title = 'Confusion Matrix')


# In[42]:


from xgboost import XGBRegressor  


# In[45]:


xgb = XGBClassifier()
from pprint import pprint

print('Parameters currently in use:\n')
pprint(xgb.get_params())


# In[46]:


# Number of trees in random forest
learning_rate = [0.05, 0.10]

# Maximum number of levels in tree
max_depth = [2, 5, 8]

min_child_weight = [1, 3, 5]

colsample_bytree = [0.4, 0.7, 1]

# Create the random grid
xgb_grid = {'learning_rate': learning_rate,
               'max_depth': max_depth,
               'min_child_weight': min_child_weight,
               'colsample_bytree': colsample_bytree,
               'scale_pos_weight': [list(train_label).count(0) / list(train_label).count(1)]}
pprint(xgb_grid)


# In[47]:


from sklearn.model_selection import GridSearchCV

xgb_search = GridSearchCV(estimator = xgb, param_grid = xgb_grid, cv = 3)
xgb_search.fit(train_data, train_label)


# In[48]:


xgb_search.best_params_


# In[49]:


best_model = xgb_search.best_estimator_
best_model.fit(train_data, train_label)


# In[50]:


test_neg_data = []
test_neg_label = []
for i in range(len(test_data)):
    if test_label[i]:
        test_neg_data.append(test_data[i])
        test_neg_label.append(1)

print("accuracy: ")
print(best_model.score(test_data, test_label))
print("recall: ")
print(best_model.score(test_neg_data, test_neg_label))


# In[54]:


# Training predictions (to demonstrate overfitting)
train_xgb_predictions = best_model.predict(train_data)
train_xgb_probs = best_model.predict_proba(train_data)[:, 1]

# Testing predictions (to determine performance)
xgb_predictions = best_model.predict(test_data)
xgb_probs = best_model.predict_proba(test_data)[:, 1]

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_label, 
                                     [1 for _ in range(len(test_label))])
    baseline['precision'] = precision_score(test_label, 
                                      [1 for _ in range(len(test_label))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_label, predictions)
    results['precision'] = precision_score(test_label, predictions)
    results['roc'] = roc_auc_score(test_label, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_label, train_predictions)
    train_results['precision'] = precision_score(train_label, train_predictions)
    train_results['roc'] = roc_auc_score(train_label, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_label, [1 for _ in range(len(test_label))])
    model_fpr, model_tpr, _ = roc_curve(test_label, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); 
    plt.title('ROC Curves');
    plt.show();

evaluate_model(xgb_predictions, xgb_probs, train_xgb_predictions, train_xgb_probs)


# In[55]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

precision, recall, thresholds = precision_recall_curve(test_label, xgb_probs)
area = auc(recall, precision)

plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC=%0.2f' % area)
plt.legend(loc="upper right")
plt.show()


# In[15]:


from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Confusion matrix
cm = confusion_matrix(test_label, xgb_predictions)
plot_confusion_matrix(cm, classes = ['0', '1'],
                      title = 'Confusion Matrix')


# In[ ]:




