# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:12:24 2023

@author: USER
"""
#pip install imputets
#pip install --upgrade spyder
# from scipy import interpolate
pip install statsmodels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
#from imputets import TimeSeriesImputer
from pandas import interpolate
# Load data

# Import data
df=pd.read_excel('C:\\Users\\USER\\Desktop\\project 99\\allot_.xlsx')
df.head()
df.describe()
df.info()
df.isnull().sum()
df.isnull().any()
 

#mean= np.mean(df)


# third business momnent
#skewnwess
import seaborn as sns

sns.histplot(df=data,x='QUANTITY') # for distribution visulatization

plt.axvline(x=df['QUANTITY'].mean(), color='red')
plt.axvline(x=df['QUANTITY'].median(), color='green')
plt.axvline(x=df['QUANTITY'].mode()[0], color='blue')

plt.show()
skewness = df['QUANTITY'].skew() #
sns.distplot(df['QUANTITY'])
# mean>median so it is positive skew



# Fourth business moment
from scipy.stats import kurtosis


df_k = np.random.normal(size=1000)

# Plot the kurtosis graph using seaborn
sns.kdeplot(df_k, bw_method='silverman')

kurt = kurtosis(df_k, fisher=True)
print("Fisher's coefficient of kurtosis:", kurt)

## histogram
plt.hist(df['QUANTITY'], bins=10)

#box plot-  TO CHECK THE OUTLIERS

plt.boxplot(df['QUANTITY']) 


## TSA


import statsmodels.api as sm


# checking the data is stationary or non stationary
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['QUANTITY'])
dfoutput=pd.Series(result[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in result[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)

#test statistic is greater than the critical value, we reject the null hypothesis 
#series is not stationary


# converting non stationary data into stationary
diff = df['QUANTITY'].diff().dropna().plot()


from statsmodels.tsa.seasonal import seasonal_decompose

df['POSTING DATE'] = pd.to_datetime(df['POSTING DATE'])
df = df.set_index('POSTING DATE').asfreq('D')
result = seasonal_decompose(df, model='ad')

decomposition = seasonal_decompose(df['QUANTITY'])
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# data preprocessing 

# droping the column not required 

df = df.drop(['EFFECTIVE DATE', 'CREATE DATE','Customer/Vendor Code','LOB','Region','BP TYPE','City','STATE','From WhsCode','From WhsName','To whsCode','TO WhsName','Model TYPE','Transfer Type','U_Frt','U_ActShipType','PRODUCT CATEGORY','ItemCode','UNIT','RATE','SO ID','SO Creation Date','SO Due Date','U_DocStatus','NumAtCard'
              ,'U_SOTYPE','BP CATEGORY','Document Type','Vehicle Type','Direct Dispatch','Comments','U_GRNNO','Loading/Unloading','Detention','KITITEM','U_AssetClass','Customer/Vendor Name'], axis=1)




##df['POSTING DATE'] = df['POSTING DATE'].dt.strftime('%d-%m-%Y')
df.info()
print(df)
# to find the outliers

sns.boxplot(x=df['QUANTITY']) #using boix plot
# Calculate using the IQR
Q1 = np.percentile(df['QUANTITY'], 25)
Q3 = np.percentile(df['QUANTITY'], 75)
IQR = Q3 - Q1

# Calculate the lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['QUANTITY'] < lower_bound) | (df['QUANTITY'] > upper_bound)]
print(outliers)   ## no outlier found


# check waether the data is sequential

df = df.sort_values('POSTING DATE')

# check if data is sequential
is_sequential = all(df['POSTING DATE'].diff().iloc[1:] > pd.Timedelta(0))

if is_sequential:
    print("Data is sequential")
else:
    print("Data is not sequential")
## I found that data is not sequential

# now converting the data into sequence
# convert date column to datetime
df['POSTING DATE'] = pd.to_datetime(df['POSTING DATE'])

# set date column as index
df.set_index('POSTING DATE', inplace=True)
# 
df = df[~df.index.duplicated(keep='first')]
df.info()
# Resample data to daily frequency
df = df.resample('w').asfreq()

# checking the missing values/nan values 
missing_values =df.isnull().sum()
print("Total number of missing values:", missing_values)
missing_values.info()
 # total 125 nan values in quantity column for weekely frequency
 
 #fill the missing/nan value with imputation and checking mape value for different imputation with barima model
  
#pip install --upgrade pandas



# Impute missing values using different methods
#mean imputation
df_mean = df.fillna(df['QUANTITY'].mean())
df_mean.isnull().sum()
df_mean.info()
#  0 null values found
#median imputation
df_median = df.fillna(df.median())

df_median.isnull().sum()

# 0 nan value found
# b fill imputation
df_bfill = df.fillna(method="bfill")

df_bfill.isnull().sum()

# 8 nan value found
# f fill imputation
df_ffill = df.fillna(method="ffill")
df_ffill.isnull().sum()
# 33 nan value found
# linear imputation
df_linear = df['QUANTITY'].interpolate()
df_linear.isnull().sum()
# 33 nan value found
# time series impuatation
#df_ts = TimeSeriesImputer().fit_transform(df)

#  Spline imputation
#df_spline = pd.DataFrame(interpolate.spline(df.index, df, np.isnan(df), order=3, axis=0), columns=df.columns)


#df_cspline = pd.DataFrame(interpolate.CubicSpline(df.index, df, axis=0), columns=df.columns)



df_mean.to_csv('df_preprocessed.csv', index=False)


# ARIMA MODEL on mean imputation 

# Split data into train and test sets
train_size = int(len(df_mean) * 0.75)
train, test = df_mean[:train_size], df_mean[train_size:]

#training data
model = ARIMA(train["QUANTITY"], order=(2, 2, 0))
model_fit = model.fit()

# Generate predictions for both training and test data
train_pred = model_fit.predict(start=train.index[0], end=train.index[-1])
test_pred = model_fit.predict(start=test.index[0], end=test.index[-1])

# Generate predictions for future data
future_pred = model_fit.forecast(steps=10)

# Evaluate performance of the model using MAPE value
train_mape = mean_absolute_error(train["QUANTITY"], train_pred) / np.mean(train["QUANTITY"]) * 100
test_mape = mean_absolute_error(test["QUANTITY"], test_pred) / np.mean(test["QUANTITY"]) * 100

print("Train MAPE:", train_mape)
print("Test MAPE:", test_mape)
# Test MAPE: 4.253495215299485



# Plot ACF and PACF GRAPH
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(train["QUANTITY"], ax=ax1)
plot_pacf(train["QUANTITY"], ax=ax2)
plt.show()

# Exponential smoothening model
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Split data into train and test sets
train_size = int(len(df_mean) * 0.75)
train, test = df_mean[:train_size], df_mean[train_size:]


# Fit exponential smoothing model to training data
model = ExponentialSmoothing(train["QUANTITY"], trend="add", seasonal="add", seasonal_periods=12)
model_fit = model.fit()

# Generate predictions for both training and test data
train_pred = model_fit.predict(start=train.index[0], end=train.index[-1])
test_pred = model_fit.predict(start=test.index[0], end=test.index[-1])

# Generate predictions for future data
future_pred = model_fit.forecast(steps=12)

# Evaluate performance of the model using MAPE value
mape_train = mean_absolute_error(train["QUANTITY"], train_pred) / np.mean(train["QUANTITY"]) * 100
mape_test = mean_absolute_error(test["QUANTITY"], test_pred) / np.mean(test["QUANTITY"]) * 100
mape_future = mean_absolute_error(future_pred, future_pred) / np.mean(future_pred) * 100
print(f"MAPE for training data: {mape_train:.2f}")
print(f"MAPE for test data: {mape_test:.2f}")
print(f"MAPE for future data: {mape_future:.2f}")

#  MAPE for test data: 8.13
#SARIMA MODEL

from statsmodels.tsa.statespace.sarimax import SARIMAX
# Split data into train and test sets
train_size = int(len(df_mean) * 0.75)
train, test = df_mean[:train_size], df_mean[train_size:]


# Fit SARIMA model to training data
model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
model_fit = model.fit()

# Generate predictions for both training and test data
train_pred = model_fit.predict(start=train.index[0], end=train.index[-1])
test_pred = model_fit.predict(start=test.index[0], end=test.index[-1])

# Generate predictions for future data
future_pred = model_fit.forecast(steps=12)

# Evaluate performance of the model using MAPE
train_mape = calc_mape(train.values, train_pred.values)
test_mape = calc_mape(test.values, test_pred.values)

print(test_mape)


# VECM MODEL
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM


# Split data into train and test sets
train_size = int(len(df_mean) * 0.75)
train, test = df_mean[:train_size], df_mean[train_size:]

# Fit VECM model to training data
model = VECM(train, k_ar_diff=1, coint_rank=1)
model_fit = model.fit()

# Generate predictions for training and test data
train_pred = model_fit.predict(start=train.index[1], end=train.index[-1])
test_pred = model_fit.predict(start=test.index[1], end=test.index[-1])

# Generate predictions for future data
future_pred = model_fit.predict(start=test.index[-1], end="2023-03-29")

# Evaluate performance of the model using MAPE
train_mape = mean_absolute_error(train.iloc[1:], train_pred) / np.mean(train.iloc[1:]) * 100
test_mape = mean_absolute_error(test.iloc[1:], test_pred) / np.mean(test.iloc[1:]) * 100

print(f"MAPE for training data: {train_mape:.2f}")
print('the test mape value :',test_mape)



#BSTS  MODEL


from pybats.analysis import analysis
from pybats.point_forecast import median


# Split data into train and test sets
train_size = int(len(df_mean) * 0.75)
train, test = df_mean[:train_size], df_mean[train_size:]

# Fit BSTS model to training data
ar_order = 1  # set AR order for local level component
dynamic_ar_order = 0  # set dynamic AR order for local level component
seasonal_period = 7  # set seasonal period to 7 days (assuming daily data)
model, samples = analysis(train["QUANTITY"], family="poisson",
                          k=ar_order, prior_length=10, rho=.5,
                          seasonal_period=seasonal_period,
                          dynamic_ar_order=dynamic_ar_order)

# Generate predictions for training and test data
train_preds = median(model.predict(samples, past_obs=train["QUANTITY"]))
test_preds = median(model.predict(samples, past_obs=test["QUANTITY"]))

# Generate predictions for future data
future_preds = median(model.predict(samples, nsim=100, forecast_horizon=30))

# Evaluate performance of model using MAPE
mape_train = mean_absolute_error(train["QUANTITY"], train_preds) / np.mean(train["QUANTITY"]) * 100
mape_test = mean_absolute_error(test["QUANTITY"], test_preds) / np.mean(test["QUANTITY"]) * 100
print(f"MAPE for training data: {mape_train:.2f}")
print(f"MAPE for test data: {mape_test:.2f}")


# AR MODEL


from statsmodels.tsa.ar_model import AutoReg



# Split data into train and test sets
train_size = int(len(df_mean) * 0.75)
train, test = df_mean[:train_size], df_mean[train_size:]


def calc_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Fit and forecast with AR model
history = train["QUANTITY"]
model_AR = AutoReg(history, lags=2)
model_fit_AR = model_AR.fit()
predictions_train = model_fit_AR.predict(start=2, end=len(train)-1)
predictions_test = model_fit_AR.predict(start=len(train), end=len(df)-1)

model_fit_AR.summary()

# Calculate MAPE for training and test sets
mape_train = calc_mape(train["QUANTITY"][2:], predictions_train)
mape_test = calc_mape(test["QUANTITY"], predictions_test)

# Print MAPE values
print(f"MAPE for training set: {mape_train:.2f}")
print(f"MAPE for test set: {mape_test:.2f}")

#MAPE for test set: 4.47
# Generate predictions for future data
future_predictions = model_fit_AR.predict(start=len(df), end=len(df)+10)

# Print future predictions
print("Future predictions:")
print(future_predictions)


import pickle
with open("AR_model_final.pkl", "wb") as f:
    pickle.dump(model_fit_AR, f)

future_predictions.to_csv('predictions.csv', index=False)
