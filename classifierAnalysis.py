# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:57:08 2021

@author: Mareike Hoffmann
"""
#README: This Code was written to find the best classifier option in the first place.
#This code is not part of the model


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB

from init import *

def classifier_analysis():
    print("start LR")
    classifier_LR = LogisticRegression(multi_class="multinomial")
    classifier_LR.fit(x_train_i,y_train_i)
    y_pred_test_LR = classifier_LR.predict(x_test_i)
    print(metrics.classification_report(y_test_i, y_pred_test_LR, digits=3))
   
    print("start XGB")
    classifier_XGB = XGBClassifier()
    classifier_XGB.fit(x_train_i,y_train_i)
    y_pred_test_XGB = classifier_XGB.predict(x_test_i)
    print(metrics.classification_report(y_test_i, y_pred_test_XGB, digits=3))
   
    print("start SVC")
    classifier_SVC = SVC(kernel="linear", probability=True)
    classifier_SVC.fit(x_train_i,y_train_i)
    y_pred_test_SVC = classifier_SVC.predict(x_test_i)
    print(metrics.classification_report(y_test_i, y_pred_test_SVC, digits=3))
   
    print("start MLP")
    classifier_MLP = MLPClassifier()
    classifier_MLP.fit(x_train_i,y_train_i)
    y_pred_test_MLP = classifier_MLP.predict(x_test_i)
    print(metrics.classification_report(y_test_i, y_pred_test_MLP, digits=3))
    
    print("start CNB")
    classifier_CNB = ComplementNB()
    classifier_CNB.fit(x_train_i,y_train_i)
    y_pred_test_CNB = classifier_CNB.predict(x_test_i)
    print(metrics.classification_report(y_test_i, y_pred_test_CNB, digits=3))
    

classifier_analysis()

   