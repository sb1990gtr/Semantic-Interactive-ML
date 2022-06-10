# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:42:22 2021

@author: Mareike Hoffmann
"""
#README: This code was written to turn the json dataset into a dataframe csv format
#Only data with a unique label and the needed attributes remained
#The needed attributes are "Volltext" and "TopicFilter", which is used as the label
#This code is not part of the model

#imports
import os
import glob
import shutil
import json
import pandas as pd

main_path = "C:/Users/makre/Documents/UniBamberg/SS2021/Masterarbeit/Masterarbeit"
dest_path = main_path + "/" + "Corpus"
src_path = main_path + "/" + "data"

try:
    os.mkdir(dest_path)
except OSError:
    print ("Creation of the directory %s failed - maybe the folder already exists" % dest_path)
else:
    print ("Successfully created the directory %s " % dest_path)
    

#search for json files in dataset folder
json_files = []
other_files = []

for file in glob.glob(src_path + '\*'):
    try:
        if file.endswith(".json"):
            json_files.append(str(file))
        else:
            other_files.append(file)
    except OSError:
        print ("No files found here!") 
print("number of json files found: %d" % len(json_files))
print("number of other files found: %d"  % len(other_files))
    

#create a list with all the unique labeled files
#check for the needed attributes in all the files
src_files=json_files
dest_files=[]

for f in src_files:
    duplicate_label=[] #temp list
    with open (f, encoding="utf8") as file:
        dataset = json.load(file)
        #check if the attributes TopicFilter and Volltext exist
        if "TopicFilter" and "Volltext" in dataset:
            #check if the dataset has mutliple labels; duplicates do not count
            try:
                for label in dataset["TopicFilter"]:
                    if label not in duplicate_label:
                        duplicate_label.append(label)
            except:
                print("file has no label")
    if len(duplicate_label) == 1:
        dest_files.append(f)


print("number of other files with unique label: %d"  % len(dest_files))


#copy the unique labeld files in the new folder
for f in dest_files:     
    shutil.copy(f, dest_path)   
    

#create a dataframe with the extracted columns from the json files and save as csv
list_of_lists = [] 

for f in glob.glob(dest_path + '\*'):
    with open (f, encoding="utf8") as file:
        dataset=json.load(file)
        list_of_lists.append([dataset["Volltext"], dataset["TopicFilter"]])
df = pd.DataFrame(list_of_lists, columns=["Text", "Label"])


#remove duplicate labels from "Label" column
j=0
for i in df["Label"]:
    df["Label"][j] = list(dict.fromkeys(i))
    j+=1

print(df)


df = df.to_csv(dest_path + '/corpus.csv')


