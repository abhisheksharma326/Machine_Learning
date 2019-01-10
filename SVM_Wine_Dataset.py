# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:58:02 2019

@author: Abhishek Sharma
"""
#importing necessary libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

#importing dataset
wine_data=pd.read_csv("E:/wine_dataset.csv")

#selecting features and target variables
x=wine_data.iloc[:,1:].values
y=wine_data.iloc[:,0:1].values

#spliting dataset for testing and training
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20) 

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