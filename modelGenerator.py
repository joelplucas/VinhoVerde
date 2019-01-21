from sys import argv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import eda.edautils as eda

def main(argv):
    file_path =  argv[1]
    wine_data = pd.read_csv(file_path, sep=';')
    processed_wine_data = preprocess_wine_data(wine_data)
    model = build_model(processed_wine_data)

def build_model(processed_wine_data):
    X_train = processed_wine_data.drop(labels=['quality'], axis=1)
    label_train = processed_wine_data['quality']

    RFC = RandomForestClassifier()
    model = RFC.fit(X_train, label_train)
    return model

def preprocess_wine_data(wine_data):
    wine_data['quality'] = pd.cut(x=wine_data['quality'], right=True, bins=(0, 6.5, 10), labels=[0, 1]).astype(int)
    eda.to_categorical(wine_data, 'type')

    drop_outliers(wine_data)
    log_transform(wine_data)
    drop_redundant_columns(wine_data)

    return wine_data


def drop_outliers(wine_data):
    errors = wine_data['alcohol'][pd.to_numeric(wine_data['alcohol'], errors='coerce').isnull()]
    wine_data.drop(errors.index, inplace=True)
    wine_data['alcohol'] = wine_data['alcohol'].astype(float)
    outliers_per_feature, outliers = eda.detect_outliers(wine_data, 2.5, wine_data.columns.values)
    wine_data.drop(outliers, inplace=True)

def log_transform(wine_data):
    wine_data['residual sugar'] = wine_data['residual sugar'].apply(np.log)
    wine_data['chlorides'] = wine_data['chlorides'].apply(np.log)
    wine_data['free sulfur dioxide'] = wine_data['free sulfur dioxide'].apply(np.log)

def drop_redundant_columns(wine_data):
    wine_data.drop(['total sulfur dioxide'], axis=1, inplace=True)
    wine_data.drop(['density'], axis=1, inplace=True)

if __name__ == "__main__":
    main(argv)