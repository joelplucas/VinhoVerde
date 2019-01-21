from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def print_PCF(data):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(data.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

def to_categorical(data, feature):
    le = preprocessing.LabelEncoder()
    le.fit(data[feature])
    data[feature] = le.transform(data[feature])

def compare_categorical_to_label(data, feature):
    print(data[[feature, 'quality']].groupby([feature]).agg(['mean', 'count']))

def plot_categorical_distribution(data, feature):
    g = sns.factorplot(x=feature, y='quality', data=data, kind="bar", size=4, palette="muted")
    g = g.set_ylabels("Wine Quality")
    g.despine(left=True)

def plot_distribution(data, feature):
    g = sns.distplot(data[feature], color="m", label="Skewness : %.2f" % (data[feature].skew()))
    g = g.legend(loc="best")

def compare_to_label(data, feature):
    g = sns.FacetGrid(data, col='quality')
    g = g.map(sns.distplot, feature)

def compare_two_features(data, featureX, featureY):
    g = sns.scatterplot(x=featureX, y=featureY, data=data)

def detect_outliers(data, iqr_factor ,features):
    outliers = []
    outliers_per_feature = {}
    for col in features:
        if type(data[col][0]) is not np.float64:
            continue
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col],75)
        IQR = Q3 - Q1
        outlier_step = iqr_factor * IQR
        feature_outliers = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step )].index
        if not feature_outliers.empty:
            outliers_per_feature[col] = feature_outliers
            outliers.extend(feature_outliers)
    return outliers_per_feature, outliers

def print_PCF(data):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(data.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)