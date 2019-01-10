# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:23:56 2019

@author: Abhishek Sharma
"""
#dataset can be downloaded from url 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/'
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#loading_dataset
wine_data=pd.read_csv("E:/wine_dataset.csv")

#normalizing the dataset
from sklearn.preprocessing import StandardScaler as sc
scaler=sc()
scaled_data=scaler.fit_transform(wine_data)
 
#selecting features and target variables
x_features=wine_data.iloc[:,1:].values
y_target=wine_data.iloc[:,0:1].values

# SKLEARN PCA : used for feature/dimentionality reduction
from sklearn.decomposition import PCA
sklearn_pca=PCA(n_components='mle')
reduced_data=sklearn_pca.fit_transform(x_features)
print(x_features.shape)
print(reduced_data.shape)

#spliting dataset for testing and training
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(reduced_data,y_target,test_size = 0.20) 

#Applying SVM on dataset
from sklearn.svm import SVC
svc_classifier=SVC(kernel='linear')
svc_classifier.fit(x_train,y_train)

#prediction on test data by the model 
y_prediction=svc_classifier.predict(x_test)

#evaluation of the result obtained after prediction
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_prediction))  
print(classification_report(y_test,y_prediction))  