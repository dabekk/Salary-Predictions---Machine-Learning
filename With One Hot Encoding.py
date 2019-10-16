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


full_set = pd.read_csv("/users/ugrad/dabekk/Downloads/Kaggle - Salary Predictions/tcd ml 2019-20 income prediction training (with labels).csv")
test_set = pd.read_csv("/users/ugrad/dabekk/Downloads/Kaggle - Salary Predictions/tcd ml 2019-20 income prediction test (without labels).csv")


# In[3]:


# one hot encode

df_cat = full_set[['Country', 'Gender', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color']]
df_dummy = pd.get_dummies(df_cat, dummy_na=True)
full_set = full_set.drop(['Country', 'Gender', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color'], axis=1)
full_set = full_set.join(df_dummy, how='outer', lsuffix='_caller', rsuffix='_other')

df_cat = test_set[['Country', 'Gender', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color']]
df_dummy = pd.get_dummies(df_cat, dummy_na=True)
test_set = test_set.drop(['Country', 'Gender', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color'], axis=1)
test_set = test_set.join(df_dummy, how='outer', lsuffix='_caller', rsuffix='_other')


print(full_set.head(3))


# In[5]:



y = full_set['Income in EUR']
X = full_set.drop(['Income in EUR'], axis=1)
print(X.shape)
print(X.head)
print(y.head())

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:



# scale data before training
print("start")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Begin Training and predictions")
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("finished")


# In[ ]:


# test_set = test_set[['Year of Record', 'Age', 'Country', 'Size of City', 'Profession', 'Body Height [cm]']]
# test_set = test_set.fillna(test_set.mean())

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# regressor = RandomForestRegressor(n_estimators=20, random_state=0)
# regressor.fit(X, y)
# y_pred = regressor.predict(test_set)

# output = pd.DataFrame({'Predicted': y_pred.flatten()})

# output.to_csv("/users/ugrad/dabekk/Downloads/Salary-Predictions---Machine-Learning-master/final_output.csv")

# print("finished")

