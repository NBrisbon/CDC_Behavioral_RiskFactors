# CDC Behavioral RiskFactors
## Behavioral Risk Factor Surveillance System

The 2017 BRFSS data continues to reflect the changes initially made in 2011 in weighting methodology (raking) and the addition of cell phone only respondents. The aggregate BRFSS combined landline and cell phone dataset is built from the landline and cell phone data submitted for 2017 and includes data for 50 states, the District of Columbia, Guam, and Puerto Rico.

There are 450,016 records for 2017.

The website is: https://www.cdc.gov/brfss/annual_data/annual_2017.html

Codebook for all variables is here: https://www.cdc.gov/brfss/annual_data/2017/pdf/codebook17_llcp-v2-508.pdf

Codebook for calculated variables is here: https://www.cdc.gov/brfss/annual_data/2017/pdf/codebook17_llcp-v2-508.pdf

## Analysis TLDR: After cleaning/exploring the data and selecting the features, I decided to use those features to predict whether or not someone would self-report any days of poor mental health over a 30-day period. Essentially, the model predicts risk for mental health problems. The features (predictors) for the final model were: gender, age, body mass index, education, alcoholic drinks per week, heavy drinker (y/n), physical activity, general health status, employment, veteran status, marital status, previous diagnosis of depression, and physical health. The final model had an accuracy score of 71%. 

## Not bad, but I suspect the model would have performed better if we were able to use the 'sleep quality' and 'life satisfaction' features. This was not possible due to excessive missing data for those variables. Oh well....

![Mental Health](behavioral-health-vs-mental-health.png)
