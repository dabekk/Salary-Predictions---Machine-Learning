from math import sqrt

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def main():
    print("Starting NLP Implementation...")
    df = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    
    df.drop(labels='Instance', axis=1, inplace=True)
    df_dummy = pd.get_dummies(df, dummy_na=True)
    print(df_dummy.head())

    #split the data into categorical, nominal, and target
    df_nominal = df_dummy.iloc[:, :5]
    df_categorical = df_dummy.iloc[:, 6:]
    df_target = df_dummy['Income in EUR']

    # scale the nominal data
    scaler = preprocessing.MinMaxScaler()
    df_nominal_scaled = scaler.fit_transform(df_nominal[['Year of Record', 'Age', 'Size of City', 'Body Height [cm]']])
    df_nominal_scaled = pd.DataFrame(df_nominal_scaled)
    df_total = pd.concat([df_nominal_scaled, df_categorical], axis=1)
    print(df_total.head(3))

    # imputation - cleaning NaNs
    df_total = df_total.fillna(df_total.mean())

    X_train, X_test, y_train, y_test = train_test_split(df_total, df_target, test_size=0.3, random_state=1)

    mlp_regressor = MLPRegressor(hidden_layer_sizes=(50, 2),
                                 activation='relu',
                                 solver='adam',
                                 learning_rate='adaptive',
                                 max_iter=10,
                                 learning_rate_init=0.01,
                                 alpha=0.01)
    # MLPRegressor -> hiddent_layer_sizes=(30,) solver='lbfgs'
    print("training model...")
    mlp_regressor.fit(X_train, y_train)
    print("Conducting predictions...")
    y_predict = mlp_regressor.predict(X_test)
    print(sqrt(mean_squared_error(y_test, y_predict)))



if __name__ == '__main__':
    main()
