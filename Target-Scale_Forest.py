#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sqrt

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# In[2]:


def target_profession(data, mean):
    
    data['Profession'] = data['Profession'].map(mean)
    return data['Profession']
    
def target_country(data, mean):
    
    data['Country'] = data['Country'].map(mean)
    return data['Country']


# In[3]:


full_set = pd.read_csv("/users/ugrad/dabekk/Downloads/Kaggle - Salary Predictions/tcd ml 2019-20 income prediction training (with labels).csv")
test_set = pd.read_csv("/users/ugrad/dabekk/Downloads/Kaggle - Salary Predictions/tcd ml 2019-20 income prediction test (without labels).csv")

medianCountry = full_set.groupby('Country')['Income in EUR'].median()
medianProf = full_set.groupby('Profession')['Income in EUR'].median()

meanCountry = full_set.groupby('Country')['Income in EUR'].mean()
meanProfession = full_set.groupby('Profession')['Income in EUR'].mean()


# In[4]:


full_set['Country'] = target_country(full_set, meanCountry)
full_set['Country'] = full_set['Country'].fillna(value=medianCountry)

test_set['Country'] = target_country(test_set, meanCountry)
test_set['Country'] = test_set['Country'].fillna(value=medianCountry)


full_set['Profession'] = target_profession(full_set, meanProfession)
full_set['Profession'] = full_set['Profession'].fillna(value=medianProf)

test_set['Profession'] = target_profession(test_set, meanProfession)
test_set['Profession'] = test_set['Profession'].fillna(value=medianProf)


print(full_set.head(3))


# In[5]:


y = full_set['Income in EUR']
X = full_set[['Year of Record', 'Age', 'Country', 'Size of City', 'Profession', 'Body Height [cm]']]
print(X.head())
print(y.head())

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[9]:


# scale data before training
print("start")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Begin Training and predictions")
regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("finished")


# In[10]:


test_set = test_set[['Year of Record', 'Age', 'Country', 'Size of City', 'Profession', 'Body Height [cm]']]
test_set = test_set.fillna(test_set.mean())

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict(test_set)

output = pd.DataFrame({'Predicted': y_pred.flatten()})

output.to_csv("/users/ugrad/dabekk/Downloads/Salary-Predictions---Machine-Learning-master/final_output_v3.csv")

print("finished")


# In[ ]:




