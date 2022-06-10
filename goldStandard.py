# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:23:25 2021

@author: Mareike Hoffmann
"""
#README: Implementation of a "gold standard" (gs) to simulate expert knowledge
#Goal is to find the most important words in each document to predict the correct label
#In real life experts could find those words by simply reading the text
#Simulation is done via feature selection
#The code section was implemented with great help from Sebastian Kiefer
import os, sys
sys.path.append(os.path.abspath("/Users/sb1990gtr/Documents/Basti/Promotion/Uni/Veröffentlichung_x/Code_Mareike/masterthesis-main/2terAnlauf/masterthesis"))

from sklearn.linear_model import LogisticRegression

from init import *

################LIME################

#LIME approach
def build_gs():
    # GS: global auf komplettem Dataset oder nur auf train? Hier wird gerade auf komplettem Dataset trainiertes Modell verwendet
    model = LogisticRegression(multi_class="auto", solver="liblinear")
    model.fit(x_data_i,y_data_i)
    pickle.dump(model, open(main_path + '/GoldStandard'+ '/gold_standard.pkl', "wb"))
    predicted = model.predict(x_data_i)
    print(classification_report(y_data_i, predicted))


#call the build_gs function once
#build_gs()

vocabulary = vec_complete.get_feature_names()
model = pickle.load(open(gs_path, "rb"))
importance_coeff = model.coef_.T
class_names = model.classes_
importance = pd.DataFrame(importance_coeff, columns=class_names, index=vocabulary)


#sort the words by importance (positive or negative) for a specific class
def provide_n_top(global_id, positive):
    class_given = y_data_i[global_id]
    if positive:
        return importance[class_given].sort_values(ascending=False)
    else:
        return importance[class_given].sort_values(ascending=True)

#LIME GOld Standard for semantic correction
def local_GS(global_id, alpha):
    #global gold standard for instance class
    words_from_gold_standard_desc = pd.DataFrame(provide_n_top(global_id, positive=True))
    words_from_gold_standard_positive = words_from_gold_standard_desc[words_from_gold_standard_desc>0].dropna()

    words_from_gold_standard_asc = pd.DataFrame(provide_n_top(global_id, positive=False))
    words_from_gold_standard_negative = words_from_gold_standard_asc[words_from_gold_standard_asc<0].dropna()

    
    #local gold standard for instance: top n words from global gs which also occur in document
    words_from_instance = pd.DataFrame(x_data[global_id].split(), columns=["word"])
    words_from_instance.index.name = 'word_index'
    relevant_words_positive = words_from_gold_standard_positive.loc[
        words_from_gold_standard_positive.index.isin(words_from_instance["word"]),][:alpha]
    relevant_words_negative = words_from_gold_standard_negative.loc[
        words_from_gold_standard_negative.index.isin(words_from_instance["word"]),][:alpha]

    #return relevant_words_positive, relevant_words_negative
    return relevant_words_positive.index.to_list(), relevant_words_negative.index.to_list()


# def local_GS_for_destructive(global_id, alpha):
#     # global gold standard for instance class
#     words_from_gold_standard_positive = pd.DataFrame(provide_n_top(global_id, positive=True))
#     words_from_gold_standard_negative = pd.DataFrame(provide_n_top(global_id, positive=False))
#
#     # local gold standard for instance: top n words from global gs which also occur in document
#     words_from_instance = pd.DataFrame(x_data[global_id].split(), columns=["word"])
#     words_from_instance.index.name = 'word_index'
#     relevant_words_positive = words_from_gold_standard_positive.loc[
#                                   words_from_gold_standard_positive.index.isin(words_from_instance["word"]),][:alpha]
#     relevant_words_negative = words_from_gold_standard_negative.loc[
#                                   words_from_gold_standard_negative.index.isin(words_from_instance["word"]),][:alpha]
#
#     # return relevant_words_positive, relevant_words_negative
#     return relevant_words_positive.index.to_list(), relevant_words_negative.index.to_list()
#

#LIME Gold Standard for Evaluation on Testset
def local_GS_eval(true_class, instance, test=False, percentage=0.05):
    words_from_gold_standard_absolute = importance[true_class].abs().sort_values(ascending=False)
    word_count = words_from_gold_standard_absolute.shape[0]
    #take the top 5% as global gold stadard for the class
    gs_proportion = round(word_count * percentage)
    top_words_from_gold_standard_absolute = words_from_gold_standard_absolute[:gs_proportion]
    
    #take all the words from the global gold standard which also occur in the document as local gold standard
    if test:
        words_from_instance = pd.DataFrame(instance.split(), columns=["word"])
    else:
        words_from_instance = pd.DataFrame(x_data[instance].split(), columns=["word"])

    words_from_instance.index.name = 'word_index'
    relevant_words_absolute = top_words_from_gold_standard_absolute.loc[
        top_words_from_gold_standard_absolute.index.isin(words_from_instance["word"]),]
    return relevant_words_absolute.index.to_list()

del vocabulary, model, importance_coeff, class_names    

    
###########################TopicLIME###########################

#topicLIME approach
from gensim.models import LdaModel
from operator import itemgetter
import gensim.corpora as corpora


def build_gs_TopicLIME():
    # LDA: global auf komplettem Dataset oder nur auf train? Hier wird gerade nur auf train trainiertes LDA verwendet
    # Die Anuahl der für den GS verwendenten Doks muss auf jeden Fall identisch sein zwischen words und topics
    lda_corpus = corpora.MmCorpus(main_path + "/LDA/" + "masterthesiscorpus_complete")
    # lda_corpus_train = corpora.MmCorpus(main_path + "/LDA/" + "corpus_train")
    # lda_corpus_test = corpora.MmCorpus(main_path + "/LDA/" + "corpus_test")

    lda_model = LdaModel.load(lda_path)
    
    #apply LDA to corpus to retreive topic distributions as features
    vecs = []
    for i in range(len(lda_corpus)):
        top_topics = lda_model.get_document_topics(lda_corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[j][1] for j in range(lda_model.num_topics)]
        vecs.append(topic_vec)

    vecs = np.array(vecs)

    # vecs_train = []
    # for i in range(len(lda_corpus_train)):
    #     top_topics = lda_model.get_document_topics(lda_corpus_train[i], minimum_probability=0.0)
    #     topic_vec = [top_topics[j][1] for j in range(lda_model.num_topics)]
    #     vecs_train.append(topic_vec)
    #
    # vecs_train = np.array(vecs_train)
    #
    #
    # vecs_test = []
    # for i in range(len(lda_corpus_test)):
    #     top_topics = lda_model.get_document_topics(lda_corpus_test[i], minimum_probability=0.0)
    #     topic_vec = [top_topics[j][1] for j in range(lda_model.num_topics)]
    #     vecs_test.append(topic_vec)
    #
    # vecs_test = np.array(vecs_test)

    pickle.dump(vecs, open(main_path + '/GoldStandard'+ '/gold_standard_TL_vecs.pkl', "wb"))
    model = LogisticRegression(multi_class="auto", solver="liblinear")
    model.fit(vecs,y_data_i)
    pickle.dump(model, open(main_path + '/GoldStandard'+ '/gold_standard_TL.pkl', "wb"))
    predicted = model.predict(vecs)
    print(classification_report(y_data_i, predicted))
    
# #call the build_gs function once
# #build_gs_TopicLIME()

model_TL = pickle.load(open(gs_path_TL, "rb"))
importance_coeff_TL = model_TL.coef_.T
class_names_TL = model_TL.classes_
importance_TL = pd.DataFrame(importance_coeff_TL, columns=class_names_TL)
vecs = pickle.load(open(main_path + '/GoldStandard'+ '/gold_standard_TL_vecs.pkl', "rb"))

#provide the n most important topics (positive or negative) for a specific class
def provide_n_top_topics(class_given, positive):
    if positive:
        return importance_TL[class_given].sort_values(ascending=False)
    else:
        return importance_TL[class_given].sort_values(ascending=True)

def local_GS_TopicLIME(label, global_id):
    vec_id = indices_train.index(global_id)
    #match words from instance with words from corresponding log reg model
    topics_from_instance = pd.DataFrame(np.array(vecs[vec_id]), columns=["topic"])
    
    topics_from_gold_standard_desc = pd.DataFrame(provide_n_top_topics(class_given=label, positive=True))
    topics_from_gold_standard_positive = topics_from_gold_standard_desc[topics_from_gold_standard_desc>0].dropna()
    topics_from_gold_standard_asc = pd.DataFrame(provide_n_top_topics(class_given=label, positive=False))
    topics_from_gold_standard_negative = topics_from_gold_standard_asc[topics_from_gold_standard_asc<0].dropna()
   
    topics_from_instance.index.name='topic_index'
    
    relevant_topics_positive = topics_from_gold_standard_positive.loc[topics_from_gold_standard_positive.index.isin(topics_from_instance.index),]
    relevant_topics_positive_with_both_attributions = relevant_topics_positive.join(topics_from_instance)
    relevant_topics_positive_with_both_attributions_with_correct_names = relevant_topics_positive_with_both_attributions.rename(columns={label : "Gold Standard attribution", "topic": "Topic attribution on document"})
    
    relevant_topics_negative = topics_from_gold_standard_negative.loc[topics_from_gold_standard_negative.index.isin(topics_from_instance.index),]
    relevant_topics_negative_with_both_attributions = relevant_topics_negative.join(topics_from_instance)
    relevant_topics_negative_with_both_attributions_with_correct_names = relevant_topics_negative_with_both_attributions.rename(columns={label : "Gold Standard attribution", "topic": "Topic attribution on document"})
    
    return topics_from_instance, relevant_topics_positive_with_both_attributions_with_correct_names, relevant_topics_negative_with_both_attributions_with_correct_names
    

#topicLIME Gold Standard for Evaluation on Testset
def local_GS_TopicLIME_eval(true_class, instance, test=False, percentage_TL=0.2):
    model = LdaModel.load(lda_path)
    if test:
        document = instance.split()
    else:
        document = x_data[instance].split()
    
    topics_from_gold_standard_absolute = importance_TL[true_class].abs().sort_values(ascending=False)
    topic_count = topics_from_gold_standard_absolute.shape[0]
    #take the top 20% as global gold stadard for the class
    gs_proportion = round(topic_count * percentage_TL)
    top_topics_from_gold_standard_absolute = list(topics_from_gold_standard_absolute.index.values)[:gs_proportion]
    
    #get the topics for each word iof the instance in case the topic allocation is possible under a minimum probability and known by lda model
    topic_list = []
    for i in document:
        try:
            topic = model.get_term_topics(i, minimum_probability=0.0000000000000000001)
            #choose topics with highest probability and add it to list
            topic_number = max(topic, key=itemgetter(1))[0]
            topic_list.append(topic_number)
        except:
            #in case no topics can be allocated or word is not in lda corpus, append number -99 to list
            topic_list.append(-99)
 
    #take all the topics from the global gold standard which also occur in the document as local gold standard
    relevant_topics_in_document = []
    for i in top_topics_from_gold_standard_absolute:
        if i in topic_list:
            relevant_topics_in_document.append(i)

    return relevant_topics_in_document


