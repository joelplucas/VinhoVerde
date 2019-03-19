# Wine Quality Prediction 

This analysis was performed on the "Wine Quality Data Set" from UCI ( https://archive.ics.uci.edu/ml/datasets/wine+quality ).

This project implements a model that estimates Portuguese wines (red and white) quality. Such implementation can be run either through a Python program (.py file) as a script or through Jupyter notebooks.


## Exploratory Data Analysis (EDA)
The "eda" directory contains the description and the source code of the EDA made on this dataset, whose implementation is on the "EDA.ipynb" file. This notebook can also be visualised on the "EDA.html" file, where all its cells were already run. In order to perform this analysis, a set of useful functions were implemented within "edautils.py".

Data cleaning, outlier detection and feature engineering were performed along with the exploratory analisis. The "eda/EDA.ipynb" notebook is also responsible for generating a file  ("winequality_processed.csv") with the processed data, which is now able to be employed on the classification models.

## Model Definition
The "modelGenerator.py" contains the source code of the model generated for estimating wine quality. The "WineQualityModeling.ipynb" notebook was implemented to generate such model, which is also saved on a ".html" version within the same directory. Such notebook shows all steps performed towards the model definition as well as the conclusions reached within the study.

This notebook is encharged for reading the processed dataset. After reading, cross validation (with k=10) is applied to evaluate the accuracy on the provided training dataset. Initially, 5 types of classifiers implementations from sklearn were evaluated. After that, sklearn's implementation
of GridSearch was employed to adjust the parameters of the top 2 score classifiers. Within 5 iterations (k=5) several parameters values were combined, where each iteration a part of the data was separated to play the test set.

The cost function employed by the algorithm responsible for generalizing the training dataset features was also described on this notebook. . 

Besides the cross validation scores reached on the final model, a learning curve was draw to evaluate the model's generalization. This learning curve shows the model is not overfitted. Finally, the accuracy scores reached by the final model were compared against the scores reached using the original dataset (without performing any data processing).