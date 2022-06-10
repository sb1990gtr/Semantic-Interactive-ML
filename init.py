# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:29:28 2021

@author: Mareike Hoffmann
"""
#README: The init file holds the dataset in its original, vectorized and split forms
#This file is used in several other files as a common basis

import os
import glob
import shutil
import json
import pandas as pd
import csv
import numpy as np
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import *


main_path = "/Users/sb1990gtr/Documents/Basti/Promotion/Uni/Veröffentlichung_x/Code_Mareike/masterthesis-main/2terAnlauf/masterthesis"
#load preprocessed corpus
documents_prepr = pd.read_csv(main_path + "/Corpus/corpus_preprocessed_complete.csv", converters={"Text": eval})
#make sure to create the lda-model & goldstandard beforehand
#check if path names are correct
gs_path = main_path + "/GoldStandard/gold_standard.pkl"
lda_path= main_path + "/LDA/models_lda13"
gs_path_TL = main_path + '/GoldStandard/gold_standard_TL.pkl'


#create LabelEncoder
le = preprocessing.LabelEncoder()
#create TFIDF Vectorizer
vectorizer_complete= TfidfVectorizer(lowercase=False)
vectorizer= TfidfVectorizer(lowercase=False)


#split into feature and target
y_data = documents_prepr["category"]
x_data = documents_prepr["text"]
length_avg_doc = [sum(len(document.split()) for document in x_data)/len(x_data)]


# #transform list of token in document into one single string and save as numpy
# for idx, line in enumerate(x_data):
#     l = str(' '.join(line).split(','))[1:-1]
#     x_data.loc[idx] = l
#
x_data = x_data.to_numpy()

#create list of original indices as a unique identification for the data
org_indices = list(range(0, len(documents_prepr)))

#delete unnecessary variable to save memory 
del documents_prepr

#encode the target/y_data label
y_data_i = le.fit_transform(y_data)
#show the encoded labels
#print("encoded labels: " + str(le.classes_))
#list(le.inverse_transform([2, 2, 1,0,3]))

#fit vectorizer on whole dataset
vec_complete = vectorizer_complete.fit(x_data)
#vectorize
x_data_i = vec_complete.transform(x_data)

#split into train and test
x_train, x_test, y_train_i, y_test_i, indices_train, indices_test = train_test_split(x_data, y_data_i, org_indices, test_size=0.2, random_state=42, stratify=y_data_i)

#fit vectorizer on whole train set
vec = vectorizer.fit(x_train)
#vectorize the x_train and x_test
x_train_i = vec.transform(x_train)
x_test_i = vec.transform(x_test)

#split the train set in labeld and pool
x_labeled, x_pool, y_labeled, y_pool, indices_labeled, indices_pool = train_test_split(x_train_i, y_train_i, indices_train, train_size=0.0001, random_state=42, stratify=y_train_i)