#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


# In[23]:


def target_profession(data, mean):
    
    data['Profession'] = data['Profession'].map(mean)
    return data['Profession']

def target_country(data, mean):
    
    data['Country'] = data['Country'].map(mean)
    return data['Country']

# def target_degree(data, mean):

#     data['University Degree'] = data['University Degree'].map(mean)
#     return data['University Degree']


# In[24]:


full_set = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
test_set = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")

medianCountry = full_set.groupby('Country')['Income in EUR'].median()
medianProf = full_set.groupby('Profession')['Income in EUR'].median()
# medianDegree = full_set.groupby('University Degree')['Income in EUR'].median()
meanCountry = full_set.groupby('Country')['Income in EUR'].mean()
meanProfession = full_set.groupby('Profession')['Income in EUR'].mean()
# meanDegree = full_set.groupby('University Degree')['Income in EUR'].mean()


full_set['Country'] = target_country(full_set, meanCountry)
full_set['Country'] = full_set['Country'].fillna(value=medianCountry)

test_set['Country'] = target_country(test_set, meanCountry)
test_set['Country'] = test_set['Country'].fillna(value=medianCountry)


full_set['Profession'] = target_profession(full_set, meanProfession)
full_set['Profession'] = full_set['Profession'].fillna(value=medianProf)

test_set['Profession'] = target_profession(test_set, meanProfession)
test_set['Profession'] = test_set['Profession'].fillna(value=medianProf)

# full_set['University Degree'] = target_degree(full_set, meanDegree)
# full_set['University Degree'] = full_set['University Degree'].fillna(value=medianDegree)

# test_set['University Degree'] = target_degree(test_set, meanDegree)
# test_set['University Degree'] = test_set['University Degree'].fillna(value=medianDegree)
 


# In[25]:


# degrees = full_set[['University Degree']]
# degree_dummy = pd.get_dummies(degrees, dummy_na=True)
# full_set = full_set.drop(['University Degree'], axis=1)
# full_set = full_set.join(degree_dummy, how='outer', lsuffix='_caller', rsuffix='_other')

# test_set = test_set.drop(['University Degree'], axis=1)
# test_set = test_set.join(degree_dummy, how='outer', lsuffix='_caller', rsuffix='_other')


# gender = full_set[['Gender']]
# gender_dummy = pd.get_dummies(gender, dummy_na=True)


# In[26]:


# # one hot encode

# df_cat = full_set[['Country', 'Gender', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color']]
# df_dummy = pd.get_dummies(df_cat, dummy_na=True)
# full_set = full_set.drop(['Country', 'Gender', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color'], axis=1)
# full_set = full_set.join(df_dummy, how='outer', lsuffix='_caller', rsuffix='_other')

# df_cat = test_set[['Country', 'Gender', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color']]
# df_dummy = pd.get_dummies(df_cat, dummy_na=True)
# test_set = test_set.drop(['Country', 'Gender', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color'], axis=1)
# test_set = test_set.join(df_dummy, how='outer', lsuffix='_caller', rsuffix='_other')


# In[27]:


print(full_set.head(3))


# In[28]:


y = full_set['Income in EUR']
X = full_set[['Year of Record', 'Age', 'Country', 'Size of City', 'Profession', 'Body Height [cm]']]
print(X.head())
print(y.head())

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[29]:


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


# In[21]:


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

# # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


