# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:18:42 2021

@author: Mareike Hoffmann
"""
#README: This code is implemented to build an optimal LDA model on the whole train dataset

from init import *

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel
from gensim.test.utils import get_tmpfile
from gensim.corpora import MmCorpus

#define the path where the trained LDA should be saved
dest_path =main_path + "/LDA"

#turn document in list form
data= [elem for elem in x_train]
data = [d.split() for d in data]
#create corpus
texts = data
#create Dictionary
id2word = corpora.Dictionary(data)
#term document frequency
corp = [id2word.doc2bow(text) for text in texts]
output_fname = get_tmpfile("dtm_corpus.mm")
MmCorpus.serialize(output_fname, corp)
corpus=MmCorpus(output_fname)

#delete the unnecessary variables to save memory capacity
del corp, output_fname, texts


#train different LDA models to find the best one
def analyse_lda(min_k, max_k, dest_path_lda):
    #create a directory for the lda analysis results
    try:
        os.mkdir(dest_path_lda)
    except OSError:
        print ("Creation of the directory %s failed - maybe the folder already exists" % dest_path_lda)
    else:
        print ("Successfully created the directory %s " % dest_path_lda)
    
    cv = []
    #analyse LDA model type gensim for different topic numbers
    for j in range(min_k,max_k):
        resdict = {}
        print(j)     

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,                                                   
                                                num_topics=j, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=1000,
                                                passes=10,
                                                alpha="auto",
                                                per_word_topics=True, 
                                                )
        #analyse the cv and the umass scores
        coherence_model_lda_cv = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='c_v')
        #coherence_model_lda_umass = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='u_mass')
        
        # ldamodeldict = {"topics" : lda_model.print_topics(num_topics=j, num_words=10),
        #                  "ppxt" : lda_model.log_perplexity(corpus),
        #                  "cs_cv" : coherence_model_lda_cv.get_coherence(),
        #                  "cs_umass" : coherence_model_lda_umass.get_coherence(),  
        #                  }
        del lda_model
        
        cv.append(coherence_model_lda_cv.get_coherence())
        print(cv)
        #save the dataframe with analytic results as csv
        # resdict.update({"number of topics "+str(j) : ldamodeldict})
        # df=pd.DataFrame(resdict.items())
        # df = df.to_csv(dest_path + '/analysis_gensim__prepr_final_' + str(j) + '.csv')


#after making the decision for a lda model: build the model and save it
def build_lda(dest_path_lda, number_of_topics):
    #make a directory for the lda model
    try:
        os.mkdir(dest_path_lda)
    except OSError:
        print ("Creation of the directory %s failed - maybe the folder already exists" % dest_path_lda)
    else:
        print ("Successfully created the directory %s " % dest_path_lda)
    
    #build LDA model type gensim
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,                                                   
                                           num_topics=number_of_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=1000,
                                           passes=10,
                                           alpha="auto",
                                           per_word_topics=True,   
                                           )
    
    #save lda model
    lda_model.save(dest_path_lda + '/LDA_gensim'+ '_Model_b2.model')
    
    return lda_model,id2word


#analyse optimal topic number k
#build the lda function once
#if __name__ == "__main__":
   #analyse_lda(min_k=36, max_k=42,dest_path_lda=dest_path)
   #build_lda(dest_path_lda=dest_path, number_of_topics=53)

del data   

