import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.api as sm
import statsmodels.graphics.api as smg

# Reading in Data
bodyfat = pd.read_table("/Users/ldskimjg/Desktop/330/4.Supervisor/HW4/BodyFat.txt", sep = " ")
print(bodyfat)
bodyfat.shape

# pair plot
sns.pairplot(bodyfat)
bodyfat.corr()

# Fitting model
explan_vars = " + ".join(bodyfat.columns[1:])
explan_vars
mylm = smf.ols("brozek ~" + explan_vars, data = bodyfat).fit()
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
ntest = round(bodyfat.shape[0]/10)

for cv in range(ncv) :
    # Choose which obs to put
    testobs = np.random.choice(bodyfat.shape[0], ntest)
    
    # Split data
    testset = bodyfat.iloc[testobs,:]
    trainset = bodyfat.drop(testobs)
    
    # Using training data to fit a model
    trainlm = smf.ols("brozek ~" + explan_vars, data = trainset).fit()
    
    # Predict test set
    testpreds = trainlm.get_prediction(testset)
    
    # Calculate bias
    bias[cv] = np.mean(testpreds.predicted_mean - testset['brozek'])
    
    # Calculate RPMSE
    rpmse[cv] = np.sqrt(np.mean(np.square(testpreds.predicted_mean-testset['brozek'])))
    
    # Calculate Coverage
    pis = testpreds.conf_int(obs=True, alpha = 0.05)
    conditions_array = [pis[:,0] <= testset['brozek'].values, pis[:,1] >= testset['brozek'].values]
    cond_met = np.all(conditions_array, axis = 0)
    
    # Calculate Width
    wid[cv] = np.mean(pis[:,1] - pis[:,0])

np.mean(bias)
np.mean(rpmse)
np.mean(cond_met)
np.mean(wid)

# 95% confidence intervals
mylm.conf_int(alpha=0.05)

# 95% prediction intervalsage = 50, weight = 203, height = 67, neck = 40.2, chest = 114.8, abdom = 108.1, hip = 102.5, thigh = 61.3, knee = 41.1, ankle = 24.7, biceps = 34.1, forearm = 31, wrist = 18.3), interval="prediction", level = 0.95)
predframe = pd.DataFrame(dict(age = [50], weight = [203], height = [67], neck = [40.2], chest = [114.8], abdom = [108.1], hip = [102.5], thigh = [61.3], knee = [41.1], ankle = [24.7], biceps = [34.1], forearm = [31], wrist = [18.3]))

preds = mylm.get_prediction(predframe)

preds.summary_frame()
preds.predicted_mean
preds.conf_int(obs = True, alpha = 0.05)
