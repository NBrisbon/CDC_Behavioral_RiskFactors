#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis and Feature Selection

# ### Install libraries

# In[50]:


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


# ### Read in data file

# In[15]:


LLCP2 = pd.read_csv(r'C:\Users\Nick\Desktop\GitProjects\LLCP_Project\LLCP2.csv')
LLCP2.head()


# #### From the below summary stats output, you'll find that participants are generally older in age. Gender is evenly distributed, for the most part. More people have less than a college education and are above poverty level in income. Most rate general health as 'good'. 

# #### Regarding the target variable (mental health), it seems that most people reported no days with poor mental health over the 30-day period. We'll want to look at this more closely to see if enough people reported some amount of poor mental health days. If this number is too low, this might not be a good target variable. 

# In[16]:


LLCP2.describe()


# #### Let's look at some bar graphs for the dependent/target variables

# In[20]:


sns.distplot(LLCP2['MENTHLTH'], kde=False, bins=5);


# In[22]:


sns.distplot(LLCP2['MENTHLTH2'], kde=False, bins=5);


# In[19]:


LLCP2[['SEX','_BMI5CAT','_EDUCAG','_INCOMG','_DRNKWEK','_RFDRHV5','_PACAT1','PA1MIN_',
       'EXERANY2','_RFHLTH','_VEGESU1','_HCVU651','EMPLOY1','VETERAN3','MARITAL','ADDEPEV2','POORHLTH',
       'PHYSHLTH','MENTHLTH', 'MENTHLTH2']].corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# ### You can see by the above table, most coefficients seems fairly small, but the output below shows many highly statistically significant correlations. This is typical with very large datasets. Based on the above output, I would also be cautious with using both alcohol variables and both poor health variables together in a regression model, as their correlations are quite high and indicative of multicollinearity. In other words, they are likely measuring the same thing, roughly.

# In[24]:


pearson_coef, p_value = stats.pearsonr(LLCP2["SEX"], LLCP2["MENTHLTH"])
print("Pearson Corr for SEX and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_AGE_G"], LLCP2["MENTHLTH"])
print("Pearson Corr for _AGE_G and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_BMI5CAT"], LLCP2["MENTHLTH"])
print("Pearson Corr for _BMI5CAT and MENTHLTH is", pearson_coef, " with p =", p_value)  

pearson_coef, p_value = stats.pearsonr(LLCP2["_EDUCAG"], LLCP2["MENTHLTH"])
print("Pearson Corr for _EDUCAG and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_INCOMG"], LLCP2["MENTHLTH"])
print("Pearson Corr _INCOMG and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_DRNKWEK"], LLCP2["MENTHLTH"])
print("Pearson Corr for _DRNKWEK and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_RFDRHV5"], LLCP2["MENTHLTH"])
print("Pearson Corr for _RFDRHV5 and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_PACAT1"], LLCP2["MENTHLTH"])
print("Pearson Corr for _PACAT1 and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["PA1MIN_"], LLCP2["MENTHLTH"])
print("Pearson Corr for PA1MIN_ and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["EXERANY2"], LLCP2["MENTHLTH"])
print("Pearson Corr for EXERANY2 and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["_RFHLTH"], LLCP2["MENTHLTH"])
print("Pearson Corr for _RFHLTH and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["_VEGESU1"], LLCP2["MENTHLTH"])
print("Pearson Corr for _VEGESU1 and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["_HCVU651"], LLCP2["MENTHLTH"])
print("Pearson Corr for _HCVU651 and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["EMPLOY1"], LLCP2["MENTHLTH"])
print("Pearson Corr for EMPLOY1 and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["VETERAN3"], LLCP2["MENTHLTH"])
print("Pearson Corr for VETERAN3 and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["MARITAL"], LLCP2["MENTHLTH"])
print("Pearson Corr for MARITAL and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["ADDEPEV2"], LLCP2["MENTHLTH"])
print("Pearson Corr for ADDEPEV2 and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["POORHLTH"], LLCP2["MENTHLTH"])
print("Pearson Corr for POORHLTH and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["PHYSHLTH"], LLCP2["MENTHLTH"])
print("Pearson Corr for PHYSHLTH and MENTHLTH is", pearson_coef, " with p =", p_value)


# ## Feature Selection

# ### 1. Filter method using a heat map and correlations

# In[25]:


plt.figure(figsize=(22,15))
cor = LLCP2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# #### Now let's only retain variables with correlations >1.0, since most relationship magnitudes are small, though significant statistically.

# In[68]:


cor_target = abs(cor["MENTHLTH"])   #Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features


# In[28]:


cor_target = abs(cor["MENTHLTH2"])   #Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features


# #### This method selects 9 variables to include as features for the continuous MENTHLTH and 7 for the binary MENTHLTH2

# ### 2. Wrapper method using RFE (Recursive Feature Elimination)

# In[31]:


X1 = LLCP2.drop(['MENTHLTH','MENTHLTH2'],1)   #Feature Matrix
y1 = LLCP2['MENTHLTH']          #Target Variable
y2 = LLCP2['MENTHLTH2']


# #### First, let's do it for MENTHLTH

# In[46]:


model = LinearRegression()  #Initializing RFE model
rfe = RFE(model, 10)  #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y1)  #Fitting the data to model
model.fit(X_rfe,y1)
print(rfe.support_)
print(rfe.ranking_)


# In[47]:


#no of features
nof_list=np.arange(1,12)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X1,y1, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[48]:


cols = list(X1.columns)
model = LinearRegression()       #Initializing RFE model
rfe = RFE(model, 10)             #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y1)   #Fitting the data to model
model.fit(X_rfe,y1)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# #### Now, do it for MENTHLTH2 using logistic regression

# In[56]:


model = LogisticRegression(solver='liblinear')  #Initializing RFE model
rfe = RFE(model, 10)  #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y2)  #Fitting the data to model
model.fit(X_rfe,y2)
print(rfe.support_)
print(rfe.ranking_)


# In[57]:


#no of features
nof_list=np.arange(1,12)            
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X1,y2, test_size = 0.3, random_state = 0)
    model = LogisticRegression(solver='liblinear')
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


# In[58]:


cols = list(X1.columns)
model = LogisticRegression(solver='liblinear')       #Initializing RFE model
rfe = RFE(model, 7)             #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y2)   #Fitting the data to model
model.fit(X_rfe,y2)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ### 3. Embedded Method using LASSO regression

# In[59]:


reg = LassoCV(max_iter=10000, cv=3)
reg.fit(X1, y1)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X1,y1))
coef = pd.Series(reg.coef_, index = X1.columns)


# In[60]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  
      str(sum(coef == 0)) + " variables")


# In[61]:


imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (25.0, 15.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[62]:


reg = LassoCV(max_iter=10000, cv=3)
reg.fit(X1, y2)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X1,y2))
coef = pd.Series(reg.coef_, index = X1.columns)


# In[63]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  
      str(sum(coef == 0)) + " variables")


# In[64]:


imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (25.0, 15.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[ ]:




