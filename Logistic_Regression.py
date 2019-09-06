#!/usr/bin/env python
# coding: utf-8

# # **Logistic Regression Model**

# ## We're using annual survey data from the CDC to predict risk for mental health problems. These are self-report measures. There are a number of features that we will use in order to predict whether or not someone will experience mental health problems over a particular month-long period. 

# ## Install libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# ## Read in the data

# In[2]:


pd.set_option('display.max_columns', 50)
LLCP2 = pd.read_csv(r'C:\Users\Nick\Desktop\GitProjects\LLCP_Project\LLCP2.csv')
LLCP2.head()


# ## Let's run a full logistic regression model first

# In[3]:


X1 = LLCP2[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_INCOMG','_RFDRHV5','_PACAT1','_RFHLTH','_HCVU651','EMPLOY1',
            'VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH']].values

y_logistic = LLCP2['MENTHLTH2'].values


# In[4]:


logit_model=sm.Logit(y_logistic,X1)
result=logit_model.fit()
print(result.summary2())


# #### We can see that all variables have p<.05.

# # Model fitting

# ## Create two df's: X = features - target; y = only target
# 
# ## Then do the test/train split and fit the model

# In[5]:


X = LLCP2[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_INCOMG','_RFDRHV5','_PACAT1','_RFHLTH','_HCVU651','EMPLOY1',
           'VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH']].values

y = LLCP2['MENTHLTH2'].values


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
probs = logreg.predict_proba(X_test)
probs = probs[:,1]

# describes info about train and test set 
print("Number of rows/columns in X_test dataset: ", X_test.shape) 
print("Number of rows/columns in y_test dataset: ", y_test.shape) 
print("Number of rows/columns in X_train dataset: ", X_train.shape) 
print("Number of rows/columns in y_train dataset: ", y_train.shape) 


# ## Predicting the test set results and calculating the accuracy

# In[7]:


y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg.score(X_test, y_test)))


# ## Confusion Matrix

# In[8]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# ### The risk of Type 2 error (bottom left) seems a bit larger than type 1 error (top right).

# #### True positive:     84398     _(We predicted a positive result and it was positive)_
# #### True negative:    18475     _(We predicted a negative result and it was negative)_
# #### False positive:    7892       _(We predicted a positive result and it was negative)_
# #### False negative:   24240     _(We predicted a negative result and it was positive)_

# In[9]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ### You can see above that the accuracy metrics look pretty good, at least for the 'Good Mental Health' ('0') value of the MENTHLTH target. The scores for the 'Poor Mental Health' value of '1' are much lower. This can be explained by the imbalanced nature of the data. 
# 
# #### If you recall from the previous 'Exploratory Analysis' file, the value count for MENTHLTH2 is imbalanced. There are roughly 300,000 rows with a '0' and about 150,000 with a '1'. Since the '0' indicates 'Good Mental Health' and has far more rows, it makes sense that the model predicts 'Good Mental Health' more accurately then 'Poor Mental Health'. 
# 
# #### There are ways of balancing the data, which we'll do below...

# ## Receiver Operating Characteristic (ROC) Curve

# In[10]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, probs)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure(figsize=(20,10))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print('AUC: %.3f' % logit_roc_auc)


# #### The receiver operating characteristic (ROC) curve is tool used with binary classifiers. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).

# # Dealing with unbalanced data:
# 
# ## The data is unbalanced, indicted by 2 things: 
# 
# ### (1) MENTHLTH2 value counts show twice as many '0' than '1' rows
# ### (2) The accuracy scores for the '1' values are far lower than the '0', showing the model is good at predicting 'Good Mental Health', but not 'Poor Mental Health'.
# 
# ### There are various re-sampling methods for dealing with unbalanced data. We will utilize the 'Under/Over-sampling' techniques. This technique drops rows at random from the 'majority class', or the over-represented value. In this case, the '0' rows will be dropped at random until both value's are equal. This can lead to a loss of information, if there is not enough data. Since we have almost 500,000 total rows, this should not be a significant problem. 

# ## First, re-check value counts for the target...you can see twice as many '0' values

# In[11]:


LLCP2.MENTHLTH2.value_counts().plot(kind='bar', title='Count (MENTHLTH2)');
LLCP2['MENTHLTH2'].value_counts()


# ## Let's first try Under-Sampling. Divide the classes. Now we undersample and concatenate back together. Then re-check the value counts. They are equal now, and we still have 142,679 rows to work with. 

# In[12]:


# Class count
count_class_0, count_class_1 = LLCP2.MENTHLTH2.value_counts()

# Divide by class
Good_MH = LLCP2[LLCP2['MENTHLTH2'] == 0]
Poor_MH = LLCP2[LLCP2['MENTHLTH2'] == 1]

Good_MH_under = Good_MH.sample(count_class_1)
LLCP2_under = pd.concat([Good_MH_under, Poor_MH], axis=0)

print('Random under-sampling:')
print(LLCP2_under.MENTHLTH2.value_counts())
print(LLCP2_under.shape)

LLCP2_under.MENTHLTH2.value_counts().plot(kind='bar', title='Count (MENTHLTH2)');


# In[13]:


LLCP2_under.describe()


# ## Now let's use the new df (LLCP2_under) to model build, as before.

# In[14]:


X_under = LLCP2_under[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_INCOMG','_RFDRHV5','_PACAT1','_RFHLTH','_HCVU651','EMPLOY1',
           'VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH']].values

y_under = LLCP2_under['MENTHLTH2'].values


# In[15]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_under, y_under, test_size=0.3, random_state=0)
logreg_under = LogisticRegression(solver='liblinear')
logreg_under.fit(X_train2, y_train2)
y_pred2 = logreg_under.predict(X_test2)
probs_under = logreg_under.predict_proba(X_test2)
probs_under = probs_under[:,1]

# describes info about train and test set 
print("Number of rows/columns in X_test2 dataset: ", X_test2.shape) 
print("Number of rows/columns in y_test2 dataset: ", y_test2.shape) 
print("Number of rows/columns in X_train2 dataset: ", X_train2.shape) 
print("Number of rows/columns in y_train2 dataset: ", y_train2.shape) 


# ## Model accuracy
# 
# ### We see the model accuracy score dropped a little bit, but it's still decent. Let's look closer.

# In[16]:


y_pred2 = logreg_under.predict(X_test2)

print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg_under.score(X_test2, y_test2)))


# ## Confusion Matrix

# In[17]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test2, y_pred2)
print(confusion_matrix)


# #### True positive:    33188     _(We predicted a positive result and it was positive)_
# #### True negative:    27226     _(We predicted a negative result and it was negative)_
# #### False positive:   9517      _(We predicted a positive result and it was negative)_
# #### False negative:   15677     _(We predicted a negative result and it was positive)_
# 
# #### So, this model makes more correct predictions, than not and the false negative rate seems a bit higher than the false positive

# ## Check the precision, recall, and F1 scores

# In[18]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test2,y_pred2))
print(classification_report(y_test2,y_pred2))
print(accuracy_score(y_test2, y_pred2))


# ### Above, we that our metrics have been lowered a bit after undersampling, however, the scores for 'poor mental health'  ('0) have been raised and are similar to 'good mental heath' ('1'). This is a much better and more balanced model than before the undersampling.

# In[19]:


logit_roc_auc = roc_auc_score(y_test2, probs_under)
fpr, tpr, thresholds = roc_curve(y_test2, probs_under)
plt.figure(figsize=(20,10))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print('AUC: %.3f' % logit_roc_auc)


# ### The ROC curve for this model looks pretty good

# ## Now, let's try Over-Sampling and compare. Always split the data into train/test sets BEFORE Over-Sampling. If not, you'll “bleed” information from the test set into the training of the models. We'll use SMOTE for this.

# In[20]:


from imblearn.over_sampling import SMOTE

# setting up testing and training sets
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.3, random_state=0)

sm = SMOTE(sampling_strategy='minority', random_state=0)
X_train_over, y_train_over = sm.fit_sample(X_train3, y_train3)

# describes info about train and test set 
print("Number of rows/columns in X_test3 dataset: ", X_test3.shape) 
print("Number of rows/columns in y_test3 dataset: ", y_test3.shape) 
print("Number of rows/columns in X_train3 dataset: ", X_train3.shape) 
print("Number of rows/columns in y_train3 dataset: ", y_train3.shape) 
print("Number of rows/columns in X_train_over dataset: ", X_train_over.shape) 
print("Number of rows/columns in y_train_over dataset: ", y_train_over.shape) 


# ### We can see below that there are now an equal number of occurances on the target variable now. There are over twice as many observations for 'Poor Mental Health', or '1' on the MENTHLTH2 target variable compared before the resampling.

# In[21]:


unique, counts = np.unique(y_train3, return_counts=True)
dict(zip(unique, counts))


# In[22]:


unique, counts = np.unique(y_train_over, return_counts=True)
dict(zip(unique, counts))


# ## Now, let's rerun that Logistic Regression Algo

# In[23]:


log_smote = LogisticRegression(solver='liblinear')
log_smote.fit(X_train_over, y_train_over)
smote_pred = log_smote.predict(X_test3)
smote_probs = log_smote.predict_proba(X_test3)
smote_probs = smote_probs[:,1]


# In[24]:


print(confusion_matrix(y_test3,smote_pred))
print(classification_report(y_test3,smote_pred))
print(accuracy_score(y_test3, smote_pred))


# In[25]:


logit_roc_auc = roc_auc_score(y_test3, smote_probs)
fpr, tpr, thresholds = roc_curve(y_test3, smote_probs)
plt.figure(figsize=(20,10))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
print('AUC: %.3f' % logit_roc_auc)


# ## We can see that the accuracy score for the Over-Sampled model is slightly higher at 73% vs 71% for Under-Sampling. However, this isn't the whole story because the accuracy of predicting Poor Mental Health (MENTHLTH2=1) dropped for the Over-Sampled model. The AUC was identical for both models too. Because we are trying to predict risk of poor mental health, we should consider the Under-Sampled model as the better model. 

# In[ ]:




