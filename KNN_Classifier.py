#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbors Classifier

# ## Install libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


# ## Read in the data

# In[2]:


pd.set_option('display.max_columns', 50)
LLCP2 = pd.read_csv(r'C:\Users\Nick\Desktop\GitProjects\LLCP_Project\LLCP2.csv')
LLCP2.head()


# ## Split the data into train/test sets (70/30 split)

# In[3]:


X = LLCP2[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_INCOMG','_RFDRHV5','_PACAT1','_RFHLTH','_HCVU651','EMPLOY1',
           'VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH']].values

y = LLCP2['MENTHLTH2'].values


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# describes info about train and test set 
print("Number of rows/columns in X_test dataset: ", X_test.shape) 
print("Number of rows/columns in y_test dataset: ", y_test.shape) 
print("Number of rows/columns in X_train dataset: ", X_train.shape) 
print("Number of rows/columns in y_train dataset: ", y_train.shape) 


# ## Fit the model

# In[9]:


#class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, 
    #p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)

KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(X_train, y_train)
y_pred = KNN_model.predict(X_test)
probs = KNN_model.predict_proba(X_test)
probs = probs[:,1]


# ## Print the accuracy reports and confusion matrix

# In[10]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# # Dealing with unbalanced data:
# 
# ## The data is unbalanced, indicted by two things: 
# 
# ### (1) MENTHLTH2 value counts show twice as many '0' than '1' rows
# ### (2) The accuracy scores for the '1' values are far lower than the '0', showing the model is biased. It's good at predicting 'Good Mental Health', but not 'Poor Mental Health'.
# 
# ### There are various re-sampling methods for dealing with unbalanced data. We will utilize the 'Under-sampling' technique. This technique drops rows at random from the 'majority class', or the over-represented value. In this case, the '0' rows will be dropped at random until both value's are equal. This can lead to a loss of information, if there is not enough data. Since we have almost 500,000 total rows, this should not be a significant problem. I'll be re-running this with other re-sampling methods in the future for comparison.

# In[11]:


LLCP2.MENTHLTH2.value_counts().plot(kind='bar', title='Count (MENTHLTH2)');
LLCP2['MENTHLTH2'].value_counts()


# ## First, re-check value counts for the target...you can see twice as many '0' values

# In[12]:


# Class count
count_class_0, count_class_1 = LLCP2.MENTHLTH2.value_counts()

# Divide by class
Good_MH = LLCP2[LLCP2['MENTHLTH2'] == 0]
Poor_MH = LLCP2[LLCP2['MENTHLTH2'] == 1]


# ## Now, we want to divide the target by value

# In[13]:


Good_MH_under = Good_MH.sample(count_class_1)
LLCP2_under = pd.concat([Good_MH_under, Poor_MH], axis=0)

print('Random under-sampling:')
print(LLCP2_under.MENTHLTH2.value_counts())

LLCP2_under.MENTHLTH2.value_counts().plot(kind='bar', title='Count (MENTHLTH2)');


# ### You can see above that we now have an equal amount of observations for both values of the target MENTHLTH2. We did lose a lot of information using this method, however, we still have a pretty large dataset to work with.

# # Under-Sampled Model

# ## Let's re-run the model now

# In[14]:


X_under = LLCP2_under[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_INCOMG','_RFDRHV5','_PACAT1','_RFHLTH','_HCVU651','EMPLOY1',
           'VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH']].values

y_under = LLCP2_under['MENTHLTH2'].values


# In[15]:


X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under, y_under, test_size=0.3, random_state=0)

# describes info about train and test set 
print("Number of rows/columns in X_test_under dataset: ", X_test_under.shape) 
print("Number of rows/columns in y_test_under dataset: ", y_test_under.shape) 
print("Number of rows/columns in X_train_under dataset: ", X_train_under.shape) 
print("Number of rows/columns in y_train_under dataset: ", y_train_under.shape) 


# In[16]:


unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))


# In[17]:


unique, counts = np.unique(y_train_under, return_counts=True)
dict(zip(unique, counts))


# In[18]:


KNN_under = KNeighborsClassifier(n_neighbors=3)
KNN_under.fit(X_train_under, y_train_under)
y_pred_under = KNN_under.predict(X_test_under)   #yields predicted class 0/1
probs_under = KNN_under.predict_proba(X_test_under)
probs_under = probs_under[:,1]    #yields probability of either class 0-1


# In[19]:


print(confusion_matrix(y_test_under,y_pred_under))
print(classification_report(y_test_under,y_pred_under))
print(accuracy_score(y_test_under, y_pred_under))


# #### The score is lower than the previous KNN model using the imbalanced data, however, this model shows decent results for both classes.
# 
# ### Confusion matrix shows that:
# #### True positive:    29688     _(We predicted a positive result and it was positive)_
# #### True negative:    27887     _(We predicted a negative result and it was negative)_
# #### False positive:   13017      _(We predicted a positive result and it was negative)_
# #### False negative:   15016     _(We predicted a negative result and it was positive)_
# 
# ### So, this model makes more correct predictions, than not and the false negative rate seems a bit higher than the false positive

# ## Now, let's run a ROC plot and get the area under the curve score (AUC)

# In[20]:


roc_auc = roc_auc_score(y_test_under, probs_under)
fpr, tpr, thresholds = roc_curve(y_test_under, probs_under)
plt.figure(figsize=(20,10))
plt.plot(fpr, tpr, label='KNN Classifier (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()
print('AUC: %.3f' % roc_auc)


# #### ROC for KNN Under-Sampled seems to be lower than both the Random Forest and Logistic Regression classifiers.

# # Over-Sampled Model

# ## Using SMOTE, we over-sample the minority class (MENTHLTH2 = 1) and take care to test/train split before preoceeding with re-sampling.

# In[21]:


from imblearn.over_sampling import SMOTE

# setting up testing and training sets
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.3, random_state=0)

sm = SMOTE(sampling_strategy='minority', random_state=0)
X_train_over, y_train_over = sm.fit_sample(X_train3, y_train3)

# describes info about train and test set 
print("Number of rows/columns in X_test3 dataset: ", X_test3.shape) 
print("Number of rows/columns in y_test3 dataset: ", y_test3.shape) 
print("Number of rows/columns in X_train_over dataset: ", X_train_over.shape) 
print("Number of rows/columns in y_train_over dataset: ", y_train_over.shape) 


# In[22]:


unique, counts = np.unique(y_train3, return_counts=True)
dict(zip(unique, counts))


# In[23]:


unique, counts = np.unique(y_train_over, return_counts=True)
dict(zip(unique, counts))


# ### We can see above, we have 215,047 observations for each value of the target now in the training set. 
# 
# ### Let's run another Random Forest Classifier with this Over-Sampled data and compare.

# In[24]:


KNN_over = KNeighborsClassifier(n_neighbors=3)
KNN_over.fit(X_train_under, y_train_under)
y_pred_over = KNN_over.predict(X_test3)   #yields predicted class 0/1
probs_over = KNN_over.predict_proba(X_test3)
probs_over = probs_over[:,1]    #yields probability of either class 0-1


# In[25]:


print(confusion_matrix(y_test3,y_pred_over))
print(classification_report(y_test3,y_pred_over))
print(accuracy_score(y_test3, y_pred_over))


# In[26]:


roc_auc = roc_auc_score(y_test3, probs_over)
fpr, tpr, thresholds = roc_curve(y_test3, probs_over)
plt.figure(figsize=(20,10))
plt.plot(fpr, tpr, label='KNN Classifier (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()
print('AUC: %.3f' % roc_auc)


# # We can see that this model is better than the previous two KNN models. The accuracy is up to 72%, though the precision is still quite low for the target (MENTHLTH2=1; Poor Mental Health). The AUC of .77 is similar to the Logistic Regression model, but not as good as the Random Forest Classifier. Both the Random Forest and Logistic regression Classifiers appear to perform better than KNN for this data.

# In[ ]:




