import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.api as sm

# Reading in Data
climate = pd.read_table("/Users/ldskimjg/Desktop/330/3.Climate/Climate.txt", sep = " ")
print(climate)
climate.shape

# Exploratory Data Analysis
sns.regplot(x = 'co2', y = 'globaltemp', data = climate, color = 'blue', lowess = True)
sns.set()
plt.title('Tempterature by co2')
plt.xlabel('CO2')
plt.ylabel('Temperature')

climate.corr()

# Fitting a linear model
mylm = smf.ols(formula = 'globaltemp ~ co2', data = climate).fit()
mylm.summary()
mylm.mse_resid
mylm.params
mylm.rsquared

# Plot of residuals vs. fitted values
dir(mylm)

sns.regplot(x = mylm.fittedvalues, y = mylm.resid, color = 'blue', ci = None)

sm.het_breuschpagan(mylm.resid_pearson, mylm.model.exog)

# Normality
sns.distplot(mylm.resid_pearson, kde = False)
mylm.summary()

# Hypothesis test
mylm.summary()

# 99% confidence intervals
mylm.conf_int(alpha=0.01)

# 95% confidence intervals for CO2 at 400
predframe = pd.DataFrame(dict(co2 = [400]))

preds = mylm.get_prediction(predframe)

preds.summary_frame()
preds.predicted_mean
preds.conf_int(alpha = 0.05)

# 97% prediction intervals
preds.conf_int(obs = True,alpha = 0.03)

# Add prediction intervals
predframe2 = pd.DataFrame(dict(co2 = np.linspace(300,400, num = 1000)))

preds2 = mylm.get_prediction(predframe2)

interval = preds2.conf_int(obs = True, alpha = 0.05)

sns.regplot(x = 'co2', y = 'globaltemp', data = climate, color = 'blue', ci=None)
plt.plot(predframe2.co2, interval)

# Cross Validation
ncv = 250
bias = np.repeat(np.NaN,ncv)
rpmse = np.repeat(np.NaN,ncv)
wid = np.repeat(np.NaN,ncv)
ntest = 10
for cv in range(ncv) :
    # Choose which obs to put
    testobs = np.random.choice(climate.shape[0],ntest)
    
    # Split data
    testset = climate.iloc[testobs,:]
    trainset = climate.drop(testobs)
    
    # Using training data to fit a model
    trainlm = smf.ols(formula = 'globaltemp ~ co2', data = trainset).fit()
    
    # Predict test set
    testpreds = trainlm.get_prediction(testset)
    
    # Calculate bias
    bias[cv] = np.mean(testpreds.predicted_mean-testset['globaltemp'])
    
    # Calculate RPMSE
    rpmse[cv] = np.sqrt(np.mean(np.square(testpreds.predicted_mean-testset['globaltemp'])))

    # Calculate Coverage
    pis = testpreds.conf_int(obs=True, alpha = 0.05)
    conditions_array = [pis[:,0] <= testset['globaltemp'].values, pis[:,1] >= testset['globaltemp'].values]
    cond_met = np.all(conditions_array,axis=0)
    
    # Calculate Width
    wid[cv] = np.mean(pis[:,1] - pis[:,0])

np.mean(bias)
np.mean(rpmse)
np.mean(cond_met)
np.mean(wid)
