from math import sqrt

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def main():
    print("Starting NLP Implementation...")
    df_test = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    df_actual = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
    splitAt = df_test.shape[0]
    print("split at " + str(splitAt))
    fullDF = df_test.append(df_actual, sort=False)
    print(fullDF.head(3))
    print("test : " + str(df_test.shape) + " after : " + str(fullDF.shape))
    fullDF = fullDF.iloc[:, 1:-1]
    print(fullDF.head(3))
    test, target = preprocess(fullDF)
    print("test")
    print(test.head(3))
    print("target")
    print(target.head(3))

    """
    #X_train, X_test, y_train, y_test = train_test_split(df_train_total, df_train_target, test_size=0.3, random_state=1)

    mlp_regressor = MLPRegressor(hidden_layer_sizes=(50, 2),
                                 activation='relu',
                                 solver='adam',
                                 learning_rate='adaptive',
                                 max_iter=2,
                                 learning_rate_init=0.01,
                                 alpha=0.01)

    print("training model...")
    mlp_regressor.fit(df_train_total, df_train_target)
    actual_data = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
    df_actual_total = preprocess(actual_data)


    print("Conducting predictions...")
    y_predict = mlp_regressor.predict(df_actual_total)
    #print(sqrt(mean_squared_error(y_test, y_predict)))
    """

def preprocess(df):
    print("Preprocessing data...")
    df_dummy = pd.get_dummies(df, dummy_na=True)
    print(df_dummy.head())
    df_dummy.to_csv("dummy.csv")

    # split the data into categorical, nominal, and target
    df_nominal = df_dummy.iloc[:, :5]
    df_categorical = df_dummy.iloc[:, 6:]
    print(df_categorical.head(3))

    df_target = df_dummy['Income in EUR']

    # scale the nominal data
    scaler = preprocessing.MinMaxScaler()
    df_nominal_scaled = scaler.fit_transform(df_nominal[['Year of Record', 'Age', 'Size of City', 'Body Height [cm]']])
    df_nominal_scaled = pd.DataFrame(df_nominal_scaled)

    df_total = df_nominal_scaled.join(df_categorical, how='outer')
    #df_total = pd.join([df_nominal_scaled.iloc[:,:], df_categorical.iloc[:,:]], axis=1)
    print(df_total.head(3))
    print(df_total.shape)

    # imputation - cleaning NaNs
    df_total = df_total.fillna(df_total.mean())
    print("Finished preprocessing")
    return df_total, df_target
if __name__ == '__main__':
    main()