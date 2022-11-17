import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv('emails.csv')

df.info()

df.head()

df.dtypes

df.drop(columns=['Email No.'], inplace=True) # cleaning

df.isna().sum()

df.describe()

# Separating the features and the labels
X=df.iloc[:, :df.shape[1]-1]       #Independent Variables
y=df.iloc[:, -1]                   #Dependent Variable
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

# Machine Learning models
# The following 5 models are used:
# K-Nearest Neighbors
# Linear SVM
# Polynomial SVM
# RBF SVM
# Sigmoid SVM
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=2),
    "Linear SVM":LinearSVC(random_state=8, max_iter=900000),
    "Polynomical SVM":SVC(kernel="poly", degree=2, random_state=8),
    "RBF SVM":SVC(kernel="rbf", random_state=8),
    "Sigmoid SVM":SVC(kernel="sigmoid", random_state=8)
}

# Fit and predict on each model
# Each model is trained using the train set and predictions are made based on the test set. Accuracy scores are calculated for each model.
for model_name, model in models.items():
    y_pred=model.fit(X_train, y_train).predict(X_test)
    print(f"Accuracy for {model_name} model \t: {metrics.accuracy_score(y_test, y_pred)}")