

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

docs = pd.read_csv(main_path + "/Corpus/preprocessedCorpus.csv")

