import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.api as sm
import statsmodels.graphics.api as smg

# Reading in Data
supervisor = pd.read_table("/Users/ldskimjg/Desktop/330/4.Supervisor/Supervisor.txt", sep = " ")
print(supervisor)
supervisor.shape

# pair plot
sns.pairplot(supervisor)
supervisor.corr()

# Fitting model
explan_vars = " + ".join(supervisor.columns[1:])
explan_vars
mylm = smf.ols("Rating ~" + explan_vars, data = supervisor).fit()
mylm.summary()
mylm.mse_resid
mylm.params
mylm.rsquared

# Linearity - added variable plot
smg.plot_partregress_grid(mylm)

# BP test & plot of residuals vs. fitted values
sns.regplot(x = mylm.fittedvalues, y = mylm.resid, color = 'blue', ci = None)
sm.het_breuschpagan(mylm.resid_pearson, mylm.model.exog)

# Normality
sns.distplot(mylm.resid_pearson, kde = False)
mylm.summary()

# Cross Validation
ncv = 250
bias = np.repeat(np.NaN,ncv)
rpmse = np.repeat(np.NaN,ncv)
wid = np.repeat(np.NaN,ncv)
ntest = round(supervisor.shape[0]/10)

for cv in range(ncv) :
    # Choose which obs to put
    testobs = np.random.choice(supervisor.shape[0], ntest)
    
    # Split data
    testset = supervisor.iloc[testobs,:]
    trainset = supervisor.drop(testobs)
    
    # Using training data to fit a model
    trainlm = smf.ols("Rating ~" + explan_vars, data = trainset).fit()
    
    # Predict test set
    testpreds = trainlm.get_prediction(testset)
    
    # Calculate bias
    bias[cv] = np.mean(testpreds.predicted_mean - testset['Rating'])
    
    # Calculate RPMSE
    rpmse[cv] = np.sqrt(np.mean(np.square(testpreds.predicted_mean-testset['Rating'])))
    
    # Calculate Coverage
    pis = testpreds.conf_int(obs=True, alpha = 0.05)
    conditions_array = [pis[:,0] <= testset['Rating'].values, pis[:,1] >= testset['Rating'].values]
    cond_met = np.all(conditions_array, axis = 0)
    
    # Calculate Width
    wid[cv] = np.mean(pis[:,1] - pis[:,0])

np.mean(bias)
np.mean(rpmse)
np.mean(cond_met)
np.mean(wid)

# 95% confidence intervals
mylm.conf_int(alpha=0.05)

# 95% prediction intervals

predframe = pd.DataFrame(dict(Complaints = [65], Privileges = [51.5], Learn = [56.5], Raises = [63.5], Critical = [77.5], Advance = [41]))

preds = mylm.get_prediction(predframe)

preds.summary_frame()
preds.predicted_mean
preds.conf_int(obs = True, alpha = 0.05)
