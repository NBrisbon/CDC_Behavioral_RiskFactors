# CDC Behavioral RiskFactors
## Behavioral Risk Factor Surveillance System

The 2017 BRFSS data continues to reflect the changes initially made in 2011 in weighting methodology (raking) and the addition of cell phone only respondents. The aggregate BRFSS combined landline and cell phone dataset is built from the landline and cell phone data submitted for 2017 and includes data for 50 states, the District of Columbia, Guam, and Puerto Rico.

There are 450,016 records for 2017.

The website is: https://www.cdc.gov/brfss/annual_data/annual_2017.html

Codebook for all variables is here: https://www.cdc.gov/brfss/annual_data/2017/pdf/codebook17_llcp-v2-508.pdf

Codebook for calculated variables is here: https://www.cdc.gov/brfss/annual_data/2017/pdf/codebook17_llcp-v2-508.pdf

## Analysis TLDR (I'll be running different models, but here we are so far): 
After cleaning/exploring the data and selecting the features, I decided to use those features to predict whether or not someone would self-report any days of poor mental health over a 30-day period. Essentially, the models predict risk for mental health problems. The features (predictors) were: gender, age, body mass index, income, education, heavy alcohol use, physical activity, general health status, employment, veteran status, marital status, previous diagnosis of depression, and physical health. I plan to run more classification and regression models. So far, the Random Forest Classifier model performed slightly better than all other models (Logistic Regression, KNN, SVM, and Gradient Boost), with an average 73% accuracy score and an area under the curve (AUC) of .81.

Not bad, considering the models are predicting mental health probelms with mainly demographic and general health features. I suspect the model would have performed even better if we were able to use the 'sleep quality' and 'life satisfaction' features. This was not possible due to excessive missing data for those variables. Oh well....

![Mental Health](behavioral-health-vs-mental-health.png)
