# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 08:53:13 2021

@author: Mareike Hoffmann
"""
#README: This code was written to preprocess the dataset and deal with imbalanced data
#This code is not part of the model

main_path = "C:/Users/makre/Documents/UniBamberg/SS2021/Masterarbeit/Masterarbeit"
nltk_datapath = "C:/Users/makre/Documents/UniBamberg/SS2021/Masterarbeit/Masterarbeit/nltkdata"
germalemma_datapath = "C:/Users/makre/Documents/UniBamberg/SS2021/Masterarbeit/Masterarbeit/germalemma"
tiger_corpus_datapath = "C:/Users/makre/Documents/UniBamberg/SS2021/Masterarbeit/Masterarbeit/germalemma/pos_tiger_trained"
dest_path = main_path + "/" + "Corpus"

import nltk 
nltk.data.path.append(nltk_datapath) 
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.utils import resample
import sys 
sys.path.append(germalemma_datapath)
from germalemma import GermaLemma
import pickle
import pandas as pd


#open the corpus dataframe and rename its columns
#doc = pd.read_csv(main_path + "/Corpus/corpus.csv")
#doc.rename(columns={"Unnamed: 0":"Index"}, inplace=True)


#load german stopwordlist and add special chars to the list
def load_stoplist(stopwords_extend= ["dass"]):
    stoplist = stopwords.words("german")
    special_char = [".",",",";","z.B.",",,","-","(",")","()",'/','  ',' ','   ',"%", ":", "+", "=", "§"]
    
    if stopwords_extend is not None:
        stoplist = stoplist + stopwords_extend
    
    stoplist_cap = [x.capitalize() for x in stoplist]
    stoplist_upper = [x.upper() for x in stoplist]
    stoplist_append = stoplist + stoplist_cap + stoplist_upper + special_char
    
    return stoplist_append


#load trained Tiger Corpus
def load_tiger_corpus(tiger_path):
    with open(tiger_path + "/" + 'tiger_based_german_classifier.pickle', 'rb') as f:
        tiger_tagger = pickle.load(f)
    return tiger_tagger

#Check POS Tags and allow only nouns, attributes, adverbs and verbs
def check_pos_tag(str_check):
    if(str_check[0][1][0:2] in ["NE","NN","VV","VA","VM","AD"]):
        return True
    else:
        return False

#turn a document into single tokens and lemmatize them if requested (lem= True or lem=False)
def tokenize_lemmatize(document, trainedcorpus, lem=True):
    tokens=[]
    for word in document:
        tokens += word_tokenize(word)
    
    #check if lemmatization should be done
    if(lem):
        lemmatizer = GermaLemma()
        lem_list = []
            
        #tag the words in the document and lemmatize them; return list with lemmatized words 
        for token in tokens:
            word_tag = trainedcorpus.tag([token])
            
            if(check_pos_tag(word_tag) == True): 
                lem_list.append(lemmatizer.find_lemma(word_tag[0][0],word_tag[0][1]))
            else:
                lem_list.append(word_tag[0][0])
            
        lem_list_low =[[x.lower() for x in lem_list]]
        return lem_list_low
    else:
        return tokens


#remove stopwords and words with less than 3 letters from the document
def remove_stopwords(tokenized_doc,stoplist):
    token_no_stw = []
    
    for text in tokenized_doc:
        for word in text:
            if word not in stoplist and len(word)>3 and word.isalpha()==True:
                token_no_stw.append(word)
    return token_no_stw
   

def preprocess_documents(src_file, dest_path, dest_filename):
    result=[]
    src_file_copy=src_file
    sl=load_stoplist()
    trainedcorpus = load_tiger_corpus(tiger_corpus_datapath)
    
    for i in range(len(src_file["Text"])):
        doc = [src_file["Text"][i]]
        tokenized_doc = tokenize_lemmatize(doc, trainedcorpus, lem=True)
        result.append(remove_stopwords(tokenized_doc, sl))

    src_file_copy["Text"]=result
    
    df = src_file_copy.to_csv(dest_path + "/" + dest_filename + '.csv')
    
    return df

#run the preprocess_documents function to preprocess all text files in the dataframe
#preprocess_documents(src_file = doc, dest_path = dest_path, dest_filename="preprocessedCorpus")
docs = pd.read_csv(main_path + "/Corpus/preprocessedCorpus.csv")

#deal with imbalanced data
def balance_Data(data):
    data = docs
    
    #delete underrepresented data
    underrepresented = ["['Reisevertrag']", "['Steuerstrafrecht']", 
                        "['Grunderwerbsteuer/Kfz-Steuer/sonstige Verkehrsteuern']",
                        "['Personalwirtschaft']", "['EDV-Recht']", 
                        "['Grundsteuer']", "['Ausländisches Recht']", 
                        "['Zollrecht']", "['EU-Recht']", 
                        "['Ausländisches Steuerrecht']", "['Steuerliche Betriebsprüfung']",
                        "['Betriebliches Rechnungswesen']", "['Neue Bundesländer']", 
                        "['Landwirtschaftsrecht']", "['Wirtschaftsprüfung']", 
                        "['Sozialversicherungsrecht']", "['ELWI-Lohn']", 
                        "['Lohnsteuer']", "['ELWI-REWE']", 
                        "['Außensteuerrecht']"]
    data2 = data[~data['Label'].isin(underrepresented)]
    
    #downsample the rest of the data to same size
    overrepresented = ["['Strafrecht']", "['Schuldverhältnisse']", 
              "['Miete Pacht']", "['Sonstiges Recht']", 
              "['Erbschaft Schenkung']", "['Erbschaftsteuer']", 
              "['Kauf Tausch Leasing']", "['Grunderwerbsteuer']", 
              "['Verbrauchsteuern']", "['Bewertung']", 
              "['SchuldrechtAT']", "['Versicherungsrecht']", 
              "['Wohnungseigentum']", "['BGBAT']", 
              "['Berufsrecht Rechtsanwälte']", "['Sachenrecht']", 
              "['Werkvertrag']", "['Gewerbesteuer']", 
              "['Insolvenzrecht']", "['Doppelbesteuerungsabkommen']", 
              "['Sonstiges Steuerrecht']", "['Urheberrecht Markenrecht Patentrecht']", 
              "['Steuerliche Förderungsgesetze']", "['Verkehrsteuern']", 
              "['Kfz-Steuer']", "['Rechnungslegung']", 
              "['Berufsrecht Steuerberater']", "['Bankrecht Kreditrecht']", "['Familienrecht']",
              "['Verfahrensrecht/Abgabenordnung']", "['Handelsrecht Gesellschaftsrecht']",
              "['Wettbewerbsrecht Kartellrecht']","['Zivilverfahrensrecht']",
              "['Körperschaftsteuer']", "['Arbeitsrecht']", "['Umsatzsteuer']", 
              "['Öffentliches Recht']","['Einkommensteuer/Lohnsteuer/Kirchensteuer']",
              "['Schadensersatz']", "['Sozialrecht']"]
    
    data3 = []
    for i in overrepresented:
        data = data2[data2["Label"]==i]
        data_downsampled= resample(data, replace=False,    # sample without replacement
                                         n_samples=1000,   # to match approximately the minority class
                                         random_state=43)  # reproducible results
        data3.append(data_downsampled)

       
    data4 = pd.concat(data3)
    
    print(data4.head())
    print(len(data4))
    data4.to_csv(dest_path + "/" + 'preprocessedCorpus_balanced3.csv')
    
    data5= data4.drop(["Unnamed: 0", "Unnamed: 0.1", "Index.1"], axis='columns')
    data5.to_csv(dest_path + "/" + 'preprocessedCorpus_balanced4.csv')
    
#run the balance function to deal with the imbalanced data
balance_Data(data=docs)