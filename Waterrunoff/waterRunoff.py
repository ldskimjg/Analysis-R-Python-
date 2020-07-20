import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.api as sm

# Reading in Data
water = pd.read_table("/Users/jungeolkim/Desktop/WD/venv/water.txt", sep = " ")
print(water)
water.shape

# Exploratory Data Analysis
sns.regplot(x = 'Precip', y = 'Runoff', data = water, color = 'blue', lowess = True)
sns.set()
plt.title('Water runoff by Precipitation')
plt.xlabel('Precipitation')
plt.ylabel('Water runoff')

water.corr()

# Fitting a linear model
mylm = smf.ols(formula = 'Runoff ~ Precip', data = water).fit()
mylm.summary()
mylm.mse_resid
mylm.params
mylm.rsquared

# Linearity
sns.regplot(x = 'Precip', y = 'Runoff', data = water, color = 'blue', ci = None)

# Plot of residuals vs. fitted values
dir(mylm)

sns.regplot(x = mylm.fittedvalues, y = mylm.resid, color = 'blue', ci = None)

sm.het_breuschpagan(mylm.resid_pearson, mylm.model.exog)

# Normality
sns.distplot(mylm.resid_pearson, kde = False)
mylm.summary()

# Hypothesis test
mylm.summary()

# Cross Validation
ncv = 250
bias = np.repeat(np.NaN,ncv)
rpmse = np.repeat(np.NaN,ncv)
wid = np.repeat(np.NaN,ncv)
ntest = 4
for cv in range(ncv) :
    # Choose which obs to put
    testobs = np.random.choice(water.shape[0],ntest)
    
    # Split data
    testset = water.iloc[testobs,:]
    trainset = water.drop(testobs)
    
    # Using training data to fit a model
    trainlm = smf.ols(formula = 'Runoff ~ Precip', data = trainset).fit()
    
    # Predict test set
    testpreds = trainlm.get_prediction(testset)
    
    # Calculate bias
    bias[cv] = np.mean(testpreds.predicted_mean-testset['Runoff'])
    
    # Calculate RPMSE
    rpmse[cv] = np.sqrt(np.mean(np.square(testpreds.predicted_mean-testset['Runoff'])))

    # Calculate Coverage
    pis = testpreds.conf_int(obs=True, alpha = 0.05)
    conditions_array = [pis[:,0] <= testset['Runoff'].values, pis[:,1] >= testset['Runoff'].values]
    cond_met = np.all(conditions_array,axis=0)
    
    # Calculate Width
    wid[cv] = np.mean(pis[:,1] - pis[:,0])

np.mean(bias)
np.mean(rpmse)
np.mean(cond_met)
np.mean(wid)

# 95% confidence intervals
mylm.conf_int(alpha=0.05)

# 95% prediction intervals for Precip at 4.5
predframe = pd.DataFrame(dict(Precip = [4.5]))

preds = mylm.get_prediction(predframe)

preds.summary_frame()
preds.predicted_mean
preds.conf_int(obs = True, alpha = 0.05)