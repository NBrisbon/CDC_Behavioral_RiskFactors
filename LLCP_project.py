#!/usr/bin/env python
# coding: utf-8

# ### Install libraries

# In[2]:


import pandas as pd
import numpy as np
get_ipython().system('pip install saspy')
import saspy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# ## Data from CDC Behavioral Risk Factor Surveillance System

# The 2017 BRFSS data continues to reflect the changes initially made in 2011 in weighting methodology (raking) and the addition of cell phone only respondents. The aggregate BRFSS combined landline and cell phone dataset is built from the landline and cell phone data submitted for 2017 and includes data for 50 states, the District of Columbia, Guam, and Puerto Rico.
# 
# There are 450,016 records for 2017.
# 
# The website is: https://www.cdc.gov/brfss/annual_data/annual_2017.html
# 
# Codebook for all variables is here: https://www.cdc.gov/brfss/annual_data/2017/pdf/codebook17_llcp-v2-508.pdf
# 
# Codebook for calculated variables is here: https://www.cdc.gov/brfss/annual_data/2017/pdf/codebook17_llcp-v2-508.pdf

# In[3]:


data = pd.read_sas(r'C:\Users\Nick\Desktop\GitProjects\LLCP2017XPT\LLCP2017.xpt', format='xport')
data.head()


# In[4]:


data.shape


# ### Variables and coding info
# 
# ___STATE__ 
# 
# (US state: 1-72; MA=#25)
# 
# ___PRACE1__ 
# 
# (race: 1=White, 2= Black, 3=Native American/Alaskan, 4=Asian, 5=Native Hawaiian/Pacific Islander, 6=other, 7=no preferred race, 8=multi-racial, but chose not to select other race, 77=don't know, 99=refused)
# 
# ___AGE_G__
# 
# (1=18-24, 2=25-34, 3=35-44, 4=45-54, 5=55-64, 6=65+)
# 
# ___BMI5CAT__
# 
# (BMI: 1= underweight, 2=normal weight, 3=overweight, 4=obese, .=don't know/refused)
# 
# _Body Mass Index_
# 
# __CHILDREN__ 
# 
# (1-87, 88=None, 99=refused/missing)
# 
# _# of children in household?_
# 
# ___EDUCAG__ 
# 
# (edu: 1=<HS, 2=HS, 3=Attended College/Tech school, 4=Graduated College/Tech, 9=don't know/missing/refused)
# 
# _Education level_
# 
# ___INCOMG__ 
# 
# (1=<15k, 2=15-25k, 3=25-35k, 4=35-50k, 5=50k+, 9=don't know/missing/refused)
# 
# _Income level_
# 
# ___DRNKWEK__ 
# 
# (0=no drinks, 1-999+=number of drinks per week, 99900=don't know/refused/missing)
# 
# _# alcoholic drinks per week?_
# 
# __DRNK3GE5__
# 
# (1-76, 88=None, 77=don't know, 99=refused)
# 
# _"Considering all types of alcoholic beverages, how many times during the past 30 days did you have 5 or more drinks for men or 4 or more drinks for women on an occasion?_
# 
# ___RFDRHV5__ 
# 
# (1=no, 2=yes, 9=don't know/missing/refused)
# 
# _Heavy alcohol consumption?_
# 
# ___PACAT1__
# 
# (1=highly active, 2=active, 3=insufficiently active, 4=inactive, 9=don't know/refused/missing)
# 
# _Physical activity level_
# 
# __PA1MIN___
# 
# (0-99999)
# 
# _"Minutes of total physical activity per week?"_
# 
# __EXERANY2__
# 
# (1=Yes, 2=No, 7=dont' know, 9=refused)
# 
# _"During the past month, other than your regular job, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?_
# 
# ___RFHLTH__
# 
# (1=good or better, 2=fair or poor, 9=don't know/missing, refused)
# 
# _General health status?_
# 
# ___VEGESU1__
# 
# (1-99998)
# 
# _"Total vegetables consumed each day?"_
# 
# __ADSLEEP__
# 
# (1-14, 88=None, 77=don't know, 99=Missing)
# 
# _"Over the last 2 weeks, how many days have you had trouble falling asleep or staying asleep or sleeping too much?"_
# 
# ___HCVU651__
# 
# (1=have, 2=don't have, 9=don't know, missing, refused)
# 
# _"Do you have access to healthcare?"_
# 
# __EMPLOY1__
# 
# (1=Employed, 2=Self-Employed, 3=Unemployed 1+yrs, 4=Unemployed <1yr, 5=Homemaker, 6=Student, 7=Retired, 8=Unable to work, 9=refused)
# 
# _Employment_
# 
# __VETERAN3__
# 
# (1=Yes, 2=No, 7=don't know, 9=refused)
# 
# _"Have you ever served on active duty in the United States Armed Forces, either in the regular military or in a National Guard or military reserve unit?"_
# 
# __MARITAL__
# 
# (1=Married, 2=Divorced, 3=Widowed, 4=Separated, 5=Never Married, 6=Coupled, not married, 9=Refused)
# 
# _"What is your marital status?"_
# 
# __MARIJANA__
# 
# (1-30, 88=none, 77=don't know, 99=missing)
# 
# _"During the past  30  days, on how many days did you use marijuana or hashish?_
# 
# __ADDEPEV2__
# 
# (1=yes, 2=no, 7=don't know, 9=refused)
# 
# _"(Ever told) you have a depressive disorder (including depression, major depression, dysthymia, or minor depression)?"_
# 
# __CIMEMLOS__
# 
# (1=Yes, 2=No, 7=don't know, 9=missing)
# 
# _"During the past 12 months, have you experienced confusion or memory loss that is happening more often or is getting worse?"_
# 
# __LSATISFY__
# 
# (1=Very Satisfied, 2=Satisfied, 3=Disatisfied, 4=Very Disatisfied, 7=don't know, 9=refused)
# 
# _"In general, how satisfied are you with your life?
# 
# __FIREARM4__
# 
# (1=Yes, 2=No, 7=don't know, 9=refused)
# 
# _"Are any firearms kept in or around your home?"_
# 
# __POORHLTH__
# 
# (1-30 days not good, 77=Unsure, 88=no days bad, 99=missing)
# 
# _"During the past 30 days, for about how many days did poor physical or mental health keep you from doing your usual activities, such as self-care, work, or recreation?"
# 
# __PHYSHLTH__
# 
# (1-30 days not good, 77=Unsure, 88=no days bad, 99=missing)
# 
# _"Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?"_
# 
# __MENTHLTH__
# 
# (1-30 days not good, 77=Unsure, 88=no days bad, 99=missing)
# 
# _"Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?"_

# ### Create subset including only the variables listed above:

# In[5]:


LLCP=data[['_STATE','_PRACE1', '_AGE_G', '_BMI5CAT', 'CHILDREN', '_EDUCAG', '_INCOMG', '_DRNKWEK', 'DRNK3GE5', 
           '_RFDRHV5', '_PACAT1', 'PA1MIN_', 'EXERANY2', '_RFHLTH', '_VEGESU1', 'ADSLEEP', '_HCVU651', 
           'EMPLOY1', 'VETERAN3', 'MARITAL', 'MARIJANA', 'ADDEPEV2', 'CIMEMLOS', 'LSATISFY', 'FIREARM4',
           'POORHLTH', 'PHYSHLTH', 'MENTHLTH']]

LLCP.head(20)


# In[6]:


LLCP.dtypes


# In[7]:


LLCP.isnull().sum()


# In[8]:


LLCP.describe()


# ### Replace all "don't know/refused/missing" values with NaN

# In[9]:


LLCP['_PRACE1'].replace(77, np.nan, inplace=True)
LLCP['_PRACE1'].replace(99, np.nan, inplace=True)


# In[10]:


LLCP['_BMI5CAT'].replace('.', np.nan, inplace=True)


# In[11]:


LLCP['CHILDREN'].replace(88, 0, inplace=True)
LLCP['CHILDREN'].replace(99, np.nan, inplace=True)


# In[12]:


LLCP.CHILDREN.unique()


# #### As seen above, someone reported having 72 children. Although not entirely impossible, this seems like a "joke response", so drop that value too. The '23' also seems high, but who knows?

# In[13]:


LLCP['CHILDREN'].replace(72, np.nan, inplace=True)


# In[14]:


LLCP['_EDUCAG'].replace(9, np.nan, inplace=True)


# In[15]:


LLCP['_INCOMG'].replace(9, np.nan, inplace=True)


# In[16]:


LLCP['_DRNKWEK'].replace(99900, np.nan, inplace=True)


# In[17]:


LLCP.DRNK3GE5.unique()


# In[18]:


LLCP['DRNK3GE5'].replace(88, 0, inplace=True)
LLCP['DRNK3GE5'].replace(77, np.nan, inplace=True)
LLCP['DRNK3GE5'].replace(99, np.nan, inplace=True)


# In[19]:


LLCP['_RFDRHV5'].replace(9, np.nan, inplace=True)


# In[20]:


LLCP['_PACAT1'].replace(9, np.nan, inplace=True)


# In[21]:


LLCP['EXERANY2'].replace(7, np.nan, inplace=True)
LLCP['EXERANY2'].replace(9, np.nan, inplace=True)


# In[22]:


LLCP['_RFHLTH'].replace(9, np.nan, inplace=True)


# In[23]:


LLCP['ADSLEEP'].replace(88, 0, inplace=True)
LLCP['ADSLEEP'].replace(77, np.nan, inplace=True)
LLCP['ADSLEEP'].replace(99, np.nan, inplace=True)


# In[24]:


LLCP['_HCVU651'].replace(9, np.nan, inplace=True)


# In[25]:


LLCP['EMPLOY1'].replace(9, np.nan, inplace=True)


# In[26]:


LLCP['VETERAN3'].replace(7, np.nan, inplace=True)
LLCP['VETERAN3'].replace(9, np.nan, inplace=True)


# In[27]:


LLCP['MARITAL'].replace(9, np.nan, inplace=True)


# In[28]:


LLCP['MARIJANA'].replace(88, 0, inplace=True)
LLCP['MARIJANA'].replace(77, np.nan, inplace=True)
LLCP['MARIJANA'].replace(99, np.nan, inplace=True)


# In[29]:


LLCP['ADDEPEV2'].replace(2, 0, inplace=True)
LLCP['ADDEPEV2'].replace(7, np.nan, inplace=True)
LLCP['ADDEPEV2'].replace(9, np.nan, inplace=True)


# In[30]:


LLCP['CIMEMLOS'].replace(7, np.nan, inplace=True)
LLCP['CIMEMLOS'].replace(9, np.nan, inplace=True)


# In[31]:


LLCP['LSATISFY'].replace(7, np.nan, inplace=True)
LLCP['LSATISFY'].replace(9, np.nan, inplace=True)


# In[32]:


LLCP['FIREARM4'].replace(7, np.nan, inplace=True)
LLCP['FIREARM4'].replace(9, np.nan, inplace=True)


# In[33]:


LLCP['POORHLTH'].replace(88, 0, inplace=True)
LLCP['POORHLTH'].replace(77, np.nan, inplace=True)
LLCP['POORHLTH'].replace(99, np.nan, inplace=True)


# In[34]:


LLCP['PHYSHLTH'].replace(88, 0, inplace=True)
LLCP['PHYSHLTH'].replace(77, np.nan, inplace=True)
LLCP['PHYSHLTH'].replace(99, np.nan, inplace=True)


# In[35]:


LLCP['MENTHLTH'].replace(88, 0, inplace=True)
LLCP['MENTHLTH'].replace(77, np.nan, inplace=True)
LLCP['MENTHLTH'].replace(99, np.nan, inplace=True)


# In[36]:


sns.distplot(LLCP['MENTHLTH'], kde=False, bins=10);


# In[37]:


plt.scatter(LLCP_NoNaN.MENTHLTH, LLCP_NoNaN.ADSLEEP, color='blue')
plt.title("Scatterplot of Mental Health vs Physical Health")
plt.xlabel("Mental Health")
plt.ylabel("Physical Health")
plt.show()


# In[38]:


#Using Pearson Correlation

plt.figure(figsize=(22,15))
cor = LLCP.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[39]:


#Correlation with output variable

cor_target = abs(cor["MENTHLTH"])   #Selecting highly correlated features
relevant_features = cor_target[cor_target>0.19]
relevant_features


# In[40]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install missingno')
import missingno as msno


# In[41]:


msno.matrix(LLCP)


# In[42]:


LLCP2 = LLCP[['_PRACE1', '_AGE_G','_BMI5CAT','CHILDREN','_EDUCAG','_INCOMG','_DRNKWEK','_RFDRHV5','_PACAT1','PA1MIN_','EXERANY2','_RFHLTH','_VEGESU1','_HCVU651','EMPLOY1','VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH','MENTHLTH']]
LLCP2.isnull().sum()


# In[43]:


msno.matrix(LLCP2)


# In[44]:


LLCP2.describe()


# In[45]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan, strategy='median', axis=0)
LLCP2[['_PRACE1','_BMI5CAT','CHILDREN','_EDUCAG','_INCOMG','_DRNKWEK','_RFDRHV5','_PACAT1','PA1MIN_','EXERANY2','_RFHLTH','_VEGESU1','_HCVU651','EMPLOY1','VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH','MENTHLTH']]=imputer.fit_transform(LLCP2[['_PRACE1','_BMI5CAT','CHILDREN','_EDUCAG','_INCOMG','_DRNKWEK','_RFDRHV5','_PACAT1','PA1MIN_','EXERANY2','_RFHLTH','_VEGESU1','_HCVU651','EMPLOY1','VETERAN3','MARITAL','ADDEPEV2','POORHLTH','PHYSHLTH','MENTHLTH']])


# In[46]:


LLCP2.describe()


# In[47]:


LLCP2.shape


# In[48]:


LLCP2[['_PRACE1','_BMI5CAT','CHILDREN','_EDUCAG','_INCOMG','_DRNKWEK','_RFDRHV5','_PACAT1','PA1MIN_',
       'EXERANY2','_RFHLTH','_VEGESU1','_HCVU651','EMPLOY1','VETERAN3','MARITAL','ADDEPEV2','POORHLTH',
       'PHYSHLTH','MENTHLTH']].corr() 


# In[49]:


from scipy import stats

pearson_coef, p_value = stats.pearsonr(LLCP2["_PRACE1"], LLCP2["MENTHLTH"])
print("Pearson Corr for _PRACE1 and MENTHLTH is", pearson_coef, " with p =", p_value)

pearson_coef, p_value = stats.pearsonr(LLCP2["_AGE_G"], LLCP2["MENTHLTH"])
print("Pearson Corr for _AGE_G and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["_BMI5CAT"], LLCP2["MENTHLTH"])
print("Pearson Corr for _BMI5CAT and MENTHLTH is", pearson_coef, " with p =", p_value) 

pearson_coef, p_value = stats.pearsonr(LLCP2["CHILDREN"], LLCP2["MENTHLTH"])
print("Pearson Corr for CHILDREN and MENTHLTH is", pearson_coef, " with p =", p_value) 

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


# In[50]:


X1 = LLCP2.drop("MENTHLTH",1)   #Feature Matrix
y1 = LLCP2["MENTHLTH"]          #Target Variable


# In[51]:


model = LinearRegression()  #Initializing RFE model
rfe = RFE(model, 21)  #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y1)  #Fitting the data to model
model.fit(X_rfe,y1)
print(rfe.support_)
print(rfe.ranking_)


# In[52]:


#no of features
nof_list=np.arange(1,14)            
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


# In[53]:


cols = list(X1.columns)
model = LinearRegression()       #Initializing RFE model
rfe = RFE(model, 13)             #Transforming data using RFE
X_rfe = rfe.fit_transform(X1,y1)   #Fitting the data to model
model.fit(X_rfe,y1)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# In[54]:


reg = LassoCV(max_iter=10000, cv=3)
reg.fit(X1, y1)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X1,y1))
coef = pd.Series(reg.coef_, index = X1.columns)


# In[55]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  
      str(sum(coef == 0)) + " variables")


# In[56]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20.0, 8.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


# In[57]:


LLCP2[['_AGE_G', '_EDUCAG','_INCOMG', 'EXERANY2','_RFHLTH', 'MARITAL', 'ADDEPEV2', 'POORHLTH',
       'PHYSHLTH','MENTHLTH']].corr()


# In[67]:


plt.scatter(LLCP2.POORHLTH, LLCP2.MENTHLTH,  color='blue')
plt.title("Scatterplot of Mindfulness vs Stress")
plt.xlabel("Mindfulness")
plt.ylabel("Stress")
plt.show()


# In[71]:


sns.regplot(x="MENTHLTH", y="POORHLTH", data=LLCP2);


# In[73]:


LLCP_sample = LLCP2.sample(5000) # This is the importante line
xdataSample, ydataSample = LLCP_sample["MENTHLTH"], LLCP_sample["POORHLTH"]

sns.regplot(x=xdataSample, y=ydataSample) 
plt.show()


# In[58]:


X = LLCP2[['ADDEPEV2','POORHLTH','PHYSHLTH']].values

y = LLCP2['MENTHLTH'].values


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[60]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[61]:


coeff_df = pd.DataFrame(regressor.coef_) 
coeff_df


# In[62]:


y_pred = regressor.predict(X_test)


# In[63]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
 
df.head(15)


# In[69]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error-MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Square Error-RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Explained Variance Score:', regressor.score(X, y))
print('R Squared:', r2_score(y_test, y_pred))


# In[ ]:




