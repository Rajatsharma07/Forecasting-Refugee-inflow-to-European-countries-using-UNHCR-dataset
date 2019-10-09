
# coding: utf-8

# ## Importing: Please convert .ipynb files to .py files to import successfully

# # Setting up environment

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import detrend
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import ccf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import Imputer
import copy
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas.tools.plotting import autocorrelation_plot
import math
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import timedelta
#from pyramid.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Please convert <b>"model_and_evaluation.ipynb"</b> to  <b>"model_and_evaluation.py"</b> to import successfully

# In[2]:


import model_and_evaluation as models


# # Evaluation Models and Optimization

# ## Grid search to find optimal p, d, q for ARMA, ARIMA, SARIMA

# In[6]:


def evaluate_model(model_name, serie, order):
    if model_name.lower() == 'sarima': 
        predictions = models.sarima_rolling_forecast(serie, order)
        
        
    elif model_name.lower() == 'arima': 
        predictions = models.arima_rolling_forecast(serie, order)
     
        
    elif model_name.lower() == 'arma':
        predictions = models.arma_rolling_forecast(serie, order)
        
        
    rmse = sqrt(mean_squared_error(serie[models.train_size:], predictions))
    return rmse



# evaluate combinations of p, d and q values for ARMA, ARIMA, SARIMA model
def grid_search(model_name, country, p_values, d_values, q_values):
    country = country.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_model(model_name, country, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('%s %s   RMSE=%.3f' % (model_name,order,rmse))
                except:
                    continue
    print('Best %s: %s RMSE=%.3f' % (model_name, best_cfg, best_score))


# ### 1.1 Grid search for ARIMA model - Austria
#  It can be observed that the results are more accurate for (p, d, q) of (4, 1, 2)

# In[33]:


p_values = range(1, 6)
d_values = range(1, 2)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('ARIMA', models.interpolated_data['Arrivals to Austria'], p_values, d_values, q_values)


# ### 1.2   Grid search for ARIMA model - Mainland Greece
#  It can be observed that the results are more accurate for (p, d, q) of (2, 1, 5)

# In[13]:


p_values = range(1, 6)
d_values = range(1, 2)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('ARIMA', models.interpolated_data['Departures to mainland Greece'], p_values, d_values, q_values)


# ### 1.3   Grid search for ARIMA model - Hungary
#  It can be observed that the results are more accurate for (p, d, q) of (4, 1, 1)

# In[7]:


p_values = range(1, 6)
d_values = range(1, 2)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('ARIMA', interpolated_data['Arrivals to Hungary'], p_values, d_values, q_values)


# ### 2.1 Grid search for ARMA model - Austria
# It can be observed that the results are more accurate for (p, d, q) of (2, 0, 5)

# In[58]:


p_values = range(1, 6)
d_values = range(0, 1)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('ARMA', models.interpolated_data['Arrivals to Austria'], p_values, d_values, q_values)


# ### 2.2  Grid search for ARMA model - Mainland Greece
# It can be observed that the results are more accurate for (p, d, q) of (2, 0, 3)

# In[12]:


p_values = range(1, 6)
d_values = range(0, 1)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('ARMA', models.interpolated_data['Departures to mainland Greece'], p_values, d_values, q_values)


# ### 2.3  Grid search for ARMA model - Hungary
# It can be observed that the results are more accurate for (p, d, q) of (1, 0, 2)

# In[15]:


p_values = range(1, 6)
d_values = range(0, 1)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('ARMA', models.interpolated_data['Arrivals to Hungary'], p_values, d_values, q_values)


# ### 3.1 Grid search for SARIMA model - Austria
# It can be observed that the results are more accurate for (p, d, q) of (3, 1, 2)

# In[35]:


p_values = range(1, 6)
d_values = range(1, 2)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('SARIMA', models.interpolated_data['Arrivals to Austria'], p_values, d_values, q_values)


# ### 3.2 Grid search for SARIMA model - Mainland Greece
# It can be observed that the results are more accurate for (p, d, q) of (5, 1, 1)

# In[14]:


p_values = range(1, 6)
d_values = range(1, 2)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('SARIMA', models.interpolated_data['Departures to mainland Greece'], p_values, d_values, q_values)


# ### 3.3 Grid search for SARIMA model - Hungary
# It can be observed that the results are more accurate for (p, d, q) of (3, 1, 1)

# In[17]:


p_values = range(1, 6)
d_values = range(1, 2)
q_values = range(1, 6)
warnings.filterwarnings("ignore")
grid_search('SARIMA', models.interpolated_data['Arrivals to Hungary'], p_values, d_values, q_values)


# ## Nested Cross validation for AR model
# The AR model which had an RMSE value of 47.91 previously is improved to RMSE of 23.06 for the <b>train_size</b> of 317

# In[79]:


def nested_cross_validation(country, split):
    
    tscv = TimeSeriesSplit(n_splits = split)
    result = []
    rmse = []
    

    for train_index, test_index in tscv.split(country):
        cv_train, cv_test = country.iloc[train_index], country.iloc[test_index]
        model = ARIMA(cv_train, order=(3,1,0))
        model_fit = model.fit(disp=-1)
        predictions = model_fit.forecast(steps=4)[0]
        true_values = cv_test.values[:4]
        result.append([sqrt(mean_squared_error(true_values, predictions)), train_index[-1], test_index[0]])
        print('RMSE: %s    Train_split_Index = %s' % (sqrt(mean_squared_error(true_values, predictions)), train_index[-1]))
            
    return result


split = 8
print ('split = %d' % (split))
result = nested_cross_validation(models.interpolated_data['Arrivals to Austria'], split)
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ## Nested cross validation for Rolling forecast models - ARMA, ARIMA, SARIMA

# In[20]:


def nested_cross_validation_diff(country, split, order, model_name):
    tscv = TimeSeriesSplit(n_splits = split)
    result = []
    rmse = []
    

    for train_index, test_index in tscv.split(country):
        cv_train, cv_test = country.iloc[train_index], country.iloc[test_index]
        predictions = list()
        history = [x for x in cv_train]
        test = [x for x in cv_test]
        
        try:

            for t in range(len(cv_test)):
                
                if (model_name.lower() == 'arma' or model_name.lower() == 'arima'):
                    model = ARIMA(history, order=order)
                    model_fit = model.fit(disp=-1)
                    yhat_f = model_fit.forecast()[0][0]
                    
                elif (model_name.lower() == 'sarima'):
                    model = sm.tsa.statespace.SARIMAX(cv_train, order=order, seasonal_order=(1,0,0,1))
                    model_fit = model.fit(disp=-1)
                    yhat_f = model_fit.forecast()[0]
               
                predictions.append(yhat_f)
                history.append(test[t])


            true_values = cv_test.values
            result.append([sqrt(mean_squared_error(true_values, predictions)), train_index[-1], test_index[0]])
            print('RMSE: %s    Train_split_Index = %s' % (sqrt(mean_squared_error(true_values, predictions)), train_index[-1]))
        
        
        except Exception as e:
            continue
        
    return result


# ### 1.1  Nested Cross validation for ARIMA model - Austria
# 
# The RMSE value is improved from 44.4 to 43.71 for a <b>train_size</b> of 312.

# In[31]:


split, order = 7, (4,1,2)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Arrivals to Austria'], split, order, 'ARIMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ### 1.2  Nested Cross validation for ARIMA model - Mainland Greece
# 
# The RMSE value is improved from 80.45 to 74.32 for a <b>train_size</b> of 312.

# In[21]:


split, order = 7, (2,1,5)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Departures to mainland Greece'], split, order, 'ARIMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ### 1.3  Nested Cross validation for ARIMA model - Hungary
# The RMSE value is improved from 15.39 to 11.61 for a <b>train_size</b> of 331. 

# In[22]:


split, order = 13, (4,1,1)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Arrivals to Hungary'], split, order, 'ARIMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ### 2.1 Nested Cross validation for ARMA model - Austria
# The RMSE value is not improved in this case.

# In[26]:


split, order = 5, (2,0,5)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Arrivals to Austria'], split, order, 'ARMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ### 2.2 Nested Cross validation for ARMA model - Mainland Greece
# The RMSE value is improved from 58.9 to 49.59 for a <b>train_size</b> of 329

# In[25]:


split, order = 12, (2,0,3)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Departures to mainland Greece'], split, order, 'ARMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ### 2.3 Nested Cross validation for ARMA model - Hungary
# The RMSE value is improved from 24.7 to 22.59 for a <b>train_size</b> of 321

# In[27]:


# Didn't change
split, order = 9, (1,0,2)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Arrivals to Hungary'], split, order, 'ARMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ### 3.1 Nested Cross validation for SARIMA model - Austria
# The RMSE value is improved from 36.5 to 33.53 for a <b>train_size</b> of 292.

# In[28]:


split, order = 10, (3,1,2)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Arrivals to Austria'], split, order, 'SARIMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ### 3.2 Nested Cross validation for SARIMA model - Mainland Greece
# The RMSE value is not improved in this case.

# In[29]:


split, order = 7, (5,1,1)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Departures to mainland Greece'], split, order, 'SARIMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# ### 3.3 Nested Cross validation for SARIMA model - Hungary
# The RMSE value is improved from 13.06 to 8.47 for a train_size of 329

# In[30]:


split, order = 12, (3,1,1)
print ('split = %d' % (split))
result = nested_cross_validation_diff(models.interpolated_data['Arrivals to Hungary'], split, order, 'SARIMA')
print ('Best split index for the model is: %s which has RMSE value of: %s' % ((min(result)[1]), (min(result)[0])))


# # Out of Sample Forecasts for next 20 days using SARIMA model

# In[16]:


def sarima_out_of_sample_rolling_forecast(country, order, train_size, no_of_days):
    
    train, test = country[0:train_size], country[train_size:]
    history = [country for country in train]
    predictions_f = list()
    ind = list()
    timeindex = models.interpolated_data['Arrivals to Austria'].index
    
    for t in range(no_of_days):
        last_date = models.interpolated_data['Arrivals to Austria'].iloc[[-1]].index
        last_date = last_date + timedelta(days=t)
        timeindex = timeindex.union(last_date)
        model = sm.tsa.statespace.SARIMAX(history, order=order, seasonal_order=(1,0,0,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        yhat = math.ceil(yhat)
        predictions_f.append(yhat)
        obs = yhat
        last_date = last_date + timedelta(days = t)
        ind.append(last_date)
        history.append(obs)

    pred = pd.DataFrame(index = timeindex, columns=['Arrivals to Austria'])
    
    # Assigning out-sampling forecasted values
    for i in range(1, no_of_days+1):
        pred['Arrivals to Austria'].iloc[-i] = predictions_f[i-1]
        
    # Assigning original previous values    
    for i in range(0, len(interpolated_data)):
        pred['Arrivals to Austria'].iloc[i] = models.interpolated_data['Arrivals to Austria'].iloc[i]

    return (pred)





no_of_days = 20      # the number of days in the future to forecast
order = (5,1,2)
train_size = len(models.interpolated_data)
print('Out of Sample Forcasts for the upcoming %d days'%no_of_days)
pred = sarima_out_of_sample_rolling_forecast(models.interpolated_data['Arrivals to Austria'], order, train_size, no_of_days)
pred[-no_of_days+1:]


# ## Plot of the count of refugees from August to October.
# 
# #### The red line indicates the original values whereas the blue line indicates the future predicted values.

# In[17]:


rcParams['figure.figsize'] = 11, 9
ax = pred[-no_of_days:].plot()
models.interpolated_data['Arrivals to Austria'][-60:].plot(ax=ax)
plt.title('The plot of (Arrivals to Austria) for the last 3 months data', fontsize = 18)
plt.ylabel('The number of refugees for Austria', fontsize = 12)
plt.xlabel('Months and year', fontsize = 12)
plt.legend(['Original values','Predicted values'])

