#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis and Feature Selection

# ### Install libraries

# In[21]:


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

# In[22]:


LLCP2 = pd.read_csv(r'C:\Users\Nick\Desktop\GitProjects\LLCP_Project\LLCP2.csv')
LLCP2.head()


# #### From the below summary stats output, you'll find that participants are generally older in age. Gender is evenly distributed, for the most part. More people have less than a college education and are above poverty level in income. Most rate general health as 'good'. 

# #### Regarding the target variable (mental health), it seems that most people reported no days with poor mental health over the 30-day period. We'll want to look at this more closely to see if enough people reported some amount of poor mental health days. If this number is too low, this might not be a good target variable. 

# In[23]:


pd.set_option('display.max_columns', 50)
LLCP2.describe()


# ### After a quick inspection, it look slike we should check PA1MIN_ for outliers, but the other variables seem ok, based on the min/max values. Let's look at a boxplot and scatterplot with the target variable.

# In[25]:


sns.boxplot(x=LLCP2['PA1MIN_'])


# In[26]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(LLCP2['MENTHLTH'], LLCP2['PA1MIN_'])
ax.set_xlabel('Days with Poor Mental Health')
ax.set_ylabel('Minutes of Physical Activity per Week')
plt.show()


# ### As you can see, the plots show a massive range of scores, with most being close to '0'. For convenience, I will simply drop this variable because we have two other variables that measure physical activity and seem to be ok. There are ways to deal with outliers, but I don't feel it's worth it for this. Especially when you consider that even the lowers bar on the graphs (20,000) is equal to almost 2 weeks of time. This doesn't make sense based on the question linked to the variable.

# In[27]:


LLCP2 = LLCP2[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_INCOMG','_RFDRHV5','_PACAT1','EXERANY2','_RFHLTH','_HCVU651','EMPLOY1','VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH','MENTHLTH','MENTHLTH2']]
LLCP2.shape


# #### Let's look at some bar graphs for the dependent/target variables

# In[11]:


sns.distplot(LLCP2['MENTHLTH'], kde=False, bins=5);


# In[12]:


sns.distplot(LLCP2['MENTHLTH2'], kde=False, bins=5);


# In[13]:


sns.set_style('whitegrid')
sns.countplot(x='MENTHLTH2',hue='SEX',data=LLCP2,palette='RdBu_r')


# In[14]:


sns.set_style('whitegrid')
sns.countplot(x='MENTHLTH2',hue='_BMI5CAT',data=LLCP2,palette='RdBu_r')


# In[15]:


sns.set_style('whitegrid')
sns.countplot(x='MENTHLTH2',hue='_EDUCAG',data=LLCP2,palette='RdBu_r')


# In[16]:


sns.set_style('whitegrid')
sns.countplot(x='_INCOMG', hue='MENTHLTH2',data=LLCP2,palette='RdBu_r')


# In[17]:


sns.set_style('whitegrid')
sns.countplot(x='MENTHLTH2',hue='MARITAL',data=LLCP2,palette='RdBu_r')


# In[28]:


LLCP2[['SEX','_BMI5CAT','_EDUCAG','_INCOMG','_RFDRHV5','_PACAT1','EXERANY2','_RFHLTH','_HCVU651','EMPLOY1',
       'VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH','MENTHLTH', 
       'MENTHLTH2']].corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# ### You can see by the above table, most coefficients seems fairly small, but the output below shows many highly statistically significant correlations. This is typical with very large datasets. I'm concerned with the correlation between EXERANY2 and PACAT1...it's very high at -.8 and indicates likely multicollinearity. We are going to drop EXERANY2 from the analysis due to its lower correlation. 

# In[32]:


LLCP2 = LLCP2[['SEX','_AGE_G','_BMI5CAT','_EDUCAG','_INCOMG','_RFDRHV5','_PACAT1','_RFHLTH','_HCVU651','EMPLOY1','VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH','MENTHLTH','MENTHLTH2']]
LLCP2.shape


# In[33]:


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

pearson_coef, p_value = stats.pearsonr(LLCP2["_RFDRHV5"], LLCP2["MENTHLTH"])
print("Pearson Corr for _RFDRHV5 and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_PACAT1"], LLCP2["MENTHLTH"])
print("Pearson Corr for _PACAT1 and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_RFHLTH"], LLCP2["MENTHLTH"])
print("Pearson Corr for _RFHLTH and MENTHLTH is", pearson_coef, " with p =", p_value)

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


# In[54]:


pearson_coef, p_value = stats.pearsonr(LLCP2["SEX"], LLCP2["MENTHLTH2"])
print("Pearson Corr for SEX and MENTHLTH2 is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_AGE_G"], LLCP2["MENTHLTH2"])
print("Pearson Corr for _AGE_G and MENTHLTH2 is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_BMI5CAT"], LLCP2["MENTHLTH2"])
print("Pearson Corr for _BMI5CAT and MENTHLTH2 is", pearson_coef, " with p =", p_value)  

pearson_coef, p_value = stats.pearsonr(LLCP2["_EDUCAG"], LLCP2["MENTHLTH2"])
print("Pearson Corr for _EDUCAG and MENTHLTH2 is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_INCOMG"], LLCP2["MENTHLTH2"])
print("Pearson Corr _INCOMG and MENTHLTH2 is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_RFDRHV5"], LLCP2["MENTHLTH2"])
print("Pearson Corr for _RFDRHV5 and MENTHLTH2 is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_PACAT1"], LLCP2["MENTHLTH2"])
print("Pearson Corr for _PACAT1 and MENTHLTH2 is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_RFHLTH"], LLCP2["MENTHLTH2"])
print("Pearson Corr for _RFHLTH and MENTHLTH2 is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["_HCVU651"], LLCP2["MENTHLTH2"])
print("Pearson Corr for _HCVU651 and MENTHLTH2 is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["EMPLOY1"], LLCP2["MENTHLTH2"])
print("Pearson Corr for EMPLOY1 and MENTHLTH2 is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["VETERAN3"], LLCP2["MENTHLTH2"])
print("Pearson Corr for VETERAN3 and MENTHLTH2 is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["MARITAL"], LLCP2["MENTHLTH2"])
print("Pearson Corr for MARITAL and MENTHLTH2 is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["ADDEPEV2"], LLCP2["MENTHLTH2"])
print("Pearson Corr for ADDEPEV2 and MENTHLTH2 is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["POORHLTH"], LLCP2["MENTHLTH2"])
print("Pearson Corr for POORHLTH and MENTHLTH2 is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["PHYSHLTH"], LLCP2["MENTHLTH2"])
print("Pearson Corr for PHYSHLTH and MENTHLTH2 is", pearson_coef, " with p = ", p_value)


# ## Feature Selection

# ### 1. Filter method using a heat map and correlations

# In[35]:


plt.figure(figsize=(22,15))
cor = LLCP2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# #### Now let's only retain variables with correlations >1.0, since most relationship magnitudes are small, though significant statistically.

# In[36]:


cor_target = abs(cor["MENTHLTH"])   #Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features


# In[37]:


cor_target = abs(cor["MENTHLTH2"])   #Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features


# #### This method selects 8 variables to include as features for the continuous MENTHLTH and 7 for the binary MENTHLTH2

# ### 2. Wrapper method using RFE (Recursive Feature Elimination)

# In[38]:


X1 = LLCP2.drop(['MENTHLTH','MENTHLTH2'],1)   #Feature Matrix
y1 = LLCP2['MENTHLTH']          #Target Variable
y2 = LLCP2['MENTHLTH2']


# #### First, let's do it for MENTHLTH

# In[39]:


model = LinearRegression()  #Initializing RFE model
rfe = RFE(model, 10)  #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y1)  #Fitting the data to model
model.fit(X_rfe,y1)
print(rfe.support_)
print(rfe.ranking_)


# In[55]:


#no of features
nof_list=np.arange(1,15)            
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


# In[56]:


cols = list(X1.columns)
model = LinearRegression()       #Initializing RFE model
rfe = RFE(model, 14)             #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y1)   #Fitting the data to model
model.fit(X_rfe,y1)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# #### Now, do it for MENTHLTH2 using logistic regression

# In[42]:


model = LogisticRegression(solver='liblinear')  #Initializing RFE model
rfe = RFE(model, 10)  #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y2)  #Fitting the data to model
model.fit(X_rfe,y2)
print(rfe.support_)
print(rfe.ranking_)


# In[57]:


#no of features
nof_list=np.arange(1,15)            
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
rfe = RFE(model, 11)             #Transforming data using RFE
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
matplotlib.rcParams['figure.figsize'] = (25.0, 20.0)
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
matplotlib.rcParams['figure.figsize'] = (25.0, 20.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# ### As you can see, it is difficult to select features for this set, given that all variables were originally chosen based on theory and previous research. All the methods shown drop different valriables. There is no consensus. These methods of feature selection would be more useful with a larger set of available features, particularly if there was no theory to use as guidance. 
# 
# ### This being the case, I'm going to use all 15 features in the models.

# In[65]:


LLCP2.to_csv(r'C:\Users\Nick\Desktop\GitProjects\LLCP_Project\LLCP2.csv',header=True, index=None)


# In[ ]:




