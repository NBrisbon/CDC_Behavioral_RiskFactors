#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classifier

# ## Install libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
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


X = LLCP2[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_DRNKWEK','_RFDRHV5','EXERANY2','_RFHLTH','EMPLOY1',
                  'VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH']].values

y = LLCP2['MENTHLTH2'].values


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## Fit the model

# In[5]:


classifier = RandomForestClassifier(n_estimators=50, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
probs = classifier.predict_proba(X_test)
probs = probs[:,1]


# ## Print the accuracy reports and confusion matrix

# In[6]:


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

# In[7]:


LLCP2.MENTHLTH2.value_counts().plot(kind='bar', title='Count (MENTHLTH2)');
LLCP2['MENTHLTH2'].value_counts()


# ## First, re-check value counts for the target...you can see twice as many '0' values

# In[8]:


# Class count
count_class_0, count_class_1 = LLCP2.MENTHLTH2.value_counts()

# Divide by class
Good_MH = LLCP2[LLCP2['MENTHLTH2'] == 0]
Poor_MH = LLCP2[LLCP2['MENTHLTH2'] == 1]


# ## Now, we want to divide the target by value

# In[9]:


Good_MH_under = Good_MH.sample(count_class_1)
LLCP2_under = pd.concat([Good_MH_under, Poor_MH], axis=0)

print('Random under-sampling:')
print(LLCP2_under.MENTHLTH2.value_counts())

LLCP2_under.MENTHLTH2.value_counts().plot(kind='bar', title='Count (MENTHLTH2)');


# ## Let's re-run the model now

# In[10]:


X = LLCP2_under[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_DRNKWEK','_RFDRHV5','EXERANY2','_RFHLTH','EMPLOY1',
                  'VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH']].values

y = LLCP2_under['MENTHLTH2'].values


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[12]:


classifier = RandomForestClassifier(n_estimators=200, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)   #yields predicted class 0/1
probs = classifier.predict_proba(X_test)
probs = probs[:,1]    #yields probability of either class 0-1


# In[13]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# #### The accuracy score here is similar, but slightly better than the score for logistic regression (71% vs 70%). The score is lower than the previous RFC model using the unbalanced data, however, this model shows decent results for both classes.
# 
# ### Confusion matrix shows that:
# #### True positive:    32382     _(We predicted a positive result and it was positive)_
# #### True negative:    28272     _(We predicted a negative result and it was negative)_
# #### False positive:   10323      _(We predicted a positive result and it was negative)_
# #### False negative:   14631     _(We predicted a negative result and it was positive)_
# 
# ### So, this model makes more correct predictions, than not and the false negative rate seems a bit higher than the false positive

# ## Now, let's run a ROC plot and get the area under the curve score (AUC)

# In[14]:


roc_auc = roc_auc_score(y_test, probs)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure(figsize=(20,10))
plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.3f)' % roc_auc)
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


# #### ROC for both the Random Forest Classifier and Logistic Regression were very similar. Both had Area Under the Curve (AUC) at .77. 

# In[ ]:



