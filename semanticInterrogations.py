
import os, sys
sys.path.append(os.path.abspath(""))
from gensim import models
import random
import ast
from collections import Counter
from operator import itemgetter
from scipy.special import rel_entr

from init import *
#LIME imports
from explainer import LIME_explainer
from goldStandard import local_GS
#TopicLIME imports
from explainer import TopicLIME_explainer
from goldStandard import local_GS_TopicLIME
from lda import id2word
from gensim.models import LdaModel

#Destructive approach LIME - 
#remove all words from training document which are falsely relevant
#a word is claimed as relevant either if it has a positiv or a negative impact on the class
def LIME_SI_dest(global_id, prediction, query_inst, learner, alpha, num_docs):
    #gs stands for "Gold Standard" and simulates expert knowledge about relevant words
    label_pred = int("".join(map(str,prediction)))
    gs_pos, gs_neg = local_GS(global_id, alpha)
    lime = LIME_explainer(global_id, label_pred, query_inst, learner, alpha)
    document= x_data[global_id].split()
    
    #transfer the words from the lime explanation into list
    lime_expl = []
    for i in lime:
        lime_expl.append(i[0])
   
    #remove all words from document, which are falsely relevant 
    false_words=list(set(lime_expl)-set(gs_pos)-set(gs_neg))

    for i in false_words:
        while i in document:
            document.remove(i)
    print("removed words: ", false_words)
    
    document = str(" ".join(document).split(','))
    documents = [document] * num_docs
    documents_i = vec.transform(documents)
    
    return documents_i


#Constructive approach LIME -
#create new training documents out of relevant words only
#a word is claimed as relevant if it has a positiv impact on the class
def LIME_SI_const(global_id, query_inst, learner, alpha, num_docs,num_words_per_doc):
    #gs stands for "Gold Standard" and simulates expert knowledge about relevant words
    gs_pos, gs_neg = local_GS(global_id, alpha)
    true_label = y_data_i[global_id]

    lime_true_class = LIME_explainer(global_id, true_label, query_inst, learner, alpha)
    
    #create new documents for training
    documents = []
    for i in range(num_docs):
        document = []
        for j in range(num_words_per_doc):
            #choose the words randomly from the gs list of relevant positive words
            word = random.choice(gs_pos)
            document.append(word)
        document = str(" ".join(document).split(','))
        documents.append(document)
    documents_i = vec.transform(documents)  
    
    return documents_i

#Destructive approach TopicLIME -
#remove all words from training document which are falsely relevant
#together with the semantically related words
#a topic is claimed as relevant either if it has a positiv or a negative impact on the class
def TopicLIME_SI_dest(global_id, prediction, query_inst, learner, alpha, num_docs):
    model = models.ldamodel.LdaModel.load(lda_path)
    label = y_data_i[global_id]
    label_pred = int("".join(map(str,prediction)))
    topic_dist, gs_pos, gs_neg = local_GS_TopicLIME(label, global_id)
    document = (x_data[global_id]).split()

    #put all relevant topics into list
    positive_topics = list(gs_pos.index.values)
    negative_topics = list(gs_neg.index.values)

    #get the topics for each word in the document in case the topic allocation is possible under a minimum probability
    topic_list = []
    for i in document:
        word_id = id2word.token2id[i]
        topic = model.get_term_topics(word_id, minimum_probability=0.0000000000000000001)
        try:
            #choose topics with highest probability and add it to list
            topic_number = max(topic, key=itemgetter(1))[0]
            topic_list.append(topic_number)
        except:
            #in case no topics can be allocated, append number -99 to list
            topic_list.append(-99)
    print("topics in document: ", topic_list)
    
    #look for all the positive and negative relevant topics which are also represented in the document
    positive_topics_in_document = []
    for i in positive_topics:
        if i in topic_list:
            positive_topics_in_document.append(i)
    
    negative_topics_in_document = []
    for i in negative_topics:
        if i in topic_list:
            negative_topics_in_document.append(i)
    
    
    #take the top n positive and top n negative relevant topics which are also represented in the document
    correct_topics = positive_topics_in_document[:alpha] + negative_topics_in_document[:alpha]
    
    topic_lime = TopicLIME_explainer(global_id, label_pred, query_inst, learner,alpha)
    
    #transfer the topics from the topiclime explanation into list
    topic_lime_expl_topics = []
    for i in topic_lime:
        if str(i[0][7]).isspace():
            topic_lime_expl_topics.append(int(i[0][8:9]))
        else:
            topic_lime_expl_topics.append(int(i[0][7:9]))
    print("topic_lime_expl_topics: ", topic_lime_expl_topics)
        
    incorrect_explained_topics = list(set(topic_lime_expl_topics) - set(correct_topics))
    print("incorrect explained topics: ", incorrect_explained_topics)
    
    #remove all the words from the document wich belong to an incorrectly explained topic
    false_words = []
    for i,j in enumerate(topic_list):
        if j in incorrect_explained_topics:
            false_words.append(document[i]) 
    
    for i in false_words:
        while i in document:
            document.remove(i)
    print("removed words: ", false_words)
    
    document_new = str(" ".join(document).split(','))
    documents = [document_new] * num_docs
    
    documents_i = vec.transform(documents)
    
    #Evaluation Kullback Leibler
    #topic attributions stays the same except for the deleted ones
    topic_doc_attribution = topic_dist["topic"].to_list()
    topic_dist_new_doc = topic_doc_attribution
    for i in incorrect_explained_topics:
        topic_dist_new_doc[i] = 0
    #calculate the new relative attributions
    sum_attributions_new_doc = sum(topic_dist_new_doc)
    topic_dist_new_doc_relative = []
    for i in topic_dist_new_doc:
        rel = i/sum_attributions_new_doc
        topic_dist_new_doc_relative.append(rel)

    q = topic_doc_attribution
    p = topic_dist_new_doc_relative
    # calculate (P || Q)
    kl_pq = sum(rel_entr(p, q))
    #print('KL(P || Q): %.3f nats' % sum(kl_pq))
    
    return documents_i, kl_pq
    
#Constructive approach
#create new training documents out of the relevant words and their associated topics
#a topic is claimed as relevant if it has a positiv impact on the class
def TopicLIME_SI_const(global_id, query_inst, learner, alpha, num_docs,num_words_per_doc, alpha_SI_create):
    from gensim.models import LdaModel
    model = models.ldamodel.LdaModel.load(lda_path)
    label = y_data_i[global_id]
    topic_dist, gs_pos, gs_neg = local_GS_TopicLIME(label, global_id)
    document = x_data[global_id].split()
    
    #put all positive relevant topics into list
    positive_topics = list(gs_pos.index.values)
    negative_topics = list(gs_neg.index.values)
    
    #get the topics for each word in the document in case the topic allocation is possible under a minimum probability
    topic_list = []
    for i in document:
        #word_id = id2word.token2id[i]
        topic = model.get_term_topics(i, minimum_probability=0.0000000000000000001)
        try:
            #choose topics with highest probability and add it to list
            topic_number = max(topic, key=itemgetter(1))[0]
            topic_list.append(topic_number)
        except:
            #in case no topics can be allocated, append number -99 to list
            topic_list.append(-99)
    print("topics in document: ", topic_list)
    
    #look for all the (positive) relevant topics which are also represented in the document
    positive_topics_in_document = []
    for i in positive_topics:
        if i in topic_list:
            positive_topics_in_document.append(i)    

    #take the top n positive relevant topics which are also represented in the document
    top_positive_topics_in_document = positive_topics_in_document[:alpha]
   
    #add the relevant gs words and their semantic related words from document to vocabulary
    vocabulary= []
    for i in top_positive_topics_in_document:
        for j,t in enumerate(topic_list):
            if t == i:
                vocabulary.append(document[j])
    vocabulary = list(set(vocabulary))
    print("vocabulary: ",vocabulary)
    
    #create new documents out of the vocabulary
    documents = []
    for i in range(num_docs):
        doc = []
        for j in range(num_words_per_doc):
            voc = random.choice(vocabulary)
            doc.append(voc)
        doc = str(" ".join(doc).split(','))
        documents.append(doc)
   
    documents_i = vec.transform(documents)
    

    #Evaluation Kullback Leibler
    #define distributions: new doc only consists of the 7 top positive topics with equal distribution
    topic_dist_new_doc = [0] * model.num_topics
    for i in top_positive_topics_in_document:
        topic_dist_new_doc[i] = 1/alpha_SI_create
    topic_doc_attribution = topic_dist["topic"].to_list()

    q = topic_doc_attribution
    p = topic_dist_new_doc
    # calculate (P || Q)
    kl_pq = sum(rel_entr(p, q))
    #print('KL(P || Q): %.3f nats' % sum(kl_pq))
        
    return documents_i, kl_pq
    
       
#Semantic Push approach - correction
#keep the original document without falsely explained words and add an extra training part
#create the training part out of the relevant gs words and the associated topics
#expert knowledge is taken into account via topic distribution
#the weight of the topics in the net training part depends on the relevance according to gs and the attribution in the document
def SI_correct(global_id, prediction, query_inst, learner, alpha, num_docs, size_new_textpart, gs_weight):
    model = models.ldamodel.LdaModel.load(lda_path)
    label = y_data_i[global_id]
    label_pred = int("".join(map(str,prediction)))
    topic_dist, gs_pos, gs_neg = local_GS_TopicLIME(label, global_id)
    document = x_data[global_id].split()
    
    #put all relevant topics into list
    positive_topics = list(gs_pos.index.values)
    negative_topics = list(gs_neg.index.values)
    
    #get the topics for each word in the document in case the topic allocation is possible under a minimum probability
    topic_list = []
    for i in document:
        word_id = id2word.token2id[i]
        topic = model.get_term_topics(word_id, minimum_probability=0.0000000000000000001)
        try:
            #choose topics with highest probability and add it to list
            topic_number = max(topic, key=itemgetter(1))[0]
            topic_list.append(topic_number)
        except:
            #in case no topics can be allocated, append number -99 to list
            topic_list.append(-99)
    print("topics in document: ", topic_list)
    
    #look for all the positive and negative relevant topics which are also represented in the document
    positive_topics_in_document = []
    for i in positive_topics:
        if i in topic_list:
            positive_topics_in_document.append(i)
    negative_topics_in_document = []
    for i in negative_topics:
        if i in topic_list:
            negative_topics_in_document.append(i)
    
    #take the top n positive relevant topics which are also represented in the document
    correct_topics = positive_topics_in_document[:alpha] + negative_topics_in_document[:alpha]
    print("correct topics: ", correct_topics)


    topic_lime = TopicLIME_explainer(global_id, label_pred, query_inst, learner, alpha)
    
    #transfer the topics from the topiclime explanation into list
    topic_lime_expl_topics = []
    positive_topics_expl = []
    negative_topics_expl = []
    for i in topic_lime:
        if i[1] >= 0:
            if str(i[0][7]).isspace():
                positive_topics_expl.append(int(i[0][8:9]))
                topic_lime_expl_topics.append(int(i[0][8:9]))
            else:
                positive_topics_expl.append(int(i[0][7:9]))
                topic_lime_expl_topics.append(int(i[0][7:9]))
        else:
            if str(i[0][7]).isspace():
                negative_topics_expl.append(int(i[0][8:9]))
                topic_lime_expl_topics.append(int(i[0][8:9]))
            else:
                negative_topics_expl.append(int(i[0][7:9]))
                topic_lime_expl_topics.append(int(i[0][7:9]))
    print("topic_lime_expl_topics: ", topic_lime_expl_topics)
    print("positive_topics_expl: ", positive_topics_expl)
    print("negative_topics_expl: ", negative_topics_expl)
    
    correct_explained_topics = list(set(topic_lime_expl_topics) & set(correct_topics)) 
    print("number of correctly explained topics: ", len(correct_explained_topics))           
    #print("number of correctly explained topics: ", counter_correct_topics)    
    incorrect_explained_topics = list(set(topic_lime_expl_topics) - set(correct_topics))
    print("incorrect explained topics: ", incorrect_explained_topics)
    #get the topics which are missing in the positive explanations or which are falsely explained as negative
    missing_positive_topics = list(set(positive_topics_in_document[:alpha]) - set(positive_topics_expl))
    print("missing positive topics: ", missing_positive_topics)
    
    
    #remove all the words from the document wich belong to an incorrectly explained topic
    false_words = []
    for i,j in enumerate(topic_list):
        if j in incorrect_explained_topics:
            false_words.append(document[i]) 
    
    for i in false_words:
        while i in document:
            document.remove(i)
    print("removed words: ", false_words)

    #in case all relevant positive topics were explained 
    #simply use the corrected original document as training document
    if len(missing_positive_topics) == 0:
        documents = []
        for m in range(num_docs):
            #take original document without falsely relevant words
            doc = document
            doc = str(" ".join(doc).split(','))
            documents.append(doc)
        print("documents og without false: ", documents)
        documents_i = vec.transform(documents)
        
        kl_pq = 0
            
        return documents_i, kl_pq
    #else add a new text part with the positive topics which were not explained as postive relevant
    #topic distribution is taken into account
    else:     
        #only leave the document topics which are missing in the positive explanations or which are falsely explained as negative
        for i,row in gs_pos.iterrows():
            if i not in missing_positive_topics:
                gs_pos.drop(i, inplace=True)
            
            
        #sample the words according to the document and gs topic distribution
        positive_topics_gs_attribution = list(gs_pos["Gold Standard attribution"].values)
        print("positive topic gs attribution: ", positive_topics_gs_attribution)
        positive_topics_doc_attribution = list(gs_pos["Topic attribution on document"].values)
        print("positive topic doc attribution: ", positive_topics_doc_attribution)
        
        #get relative values for GS and document Topic attribution
        sum_attributions_gs = sum(positive_topics_gs_attribution)
        weighting_gs = []
        for i in positive_topics_gs_attribution:
            rel = i/sum_attributions_gs
            weighting_gs.append(rel)
        print("relative gold standard attribution: ", weighting_gs)
        
        sum_attributions_doc = sum(positive_topics_doc_attribution)
        weighting_doc = []
        for i in positive_topics_doc_attribution:
            rel = i/sum_attributions_doc
            weighting_doc.append(rel)
        print("relative document attribution: ", weighting_doc)
        
        #create a weighted proportion of both attributions
        lam = gs_weight
        added_attributions = [(lam*i) + ((1-lam)*j) for i, j in zip(weighting_gs,weighting_doc)]
        print("added attributions ",added_attributions)
        
        #create a vocabulary out of the lda words from the positive relevant topics
        lda_topics= model.show_topics(num_topics=model.num_topics, num_words=40, log=False, formatted=False)
        vocabulary = []
        for i in missing_positive_topics:
            topic_words = [(lda_topics[i][0], [wd[0] for wd in lda_topics[i][1]])]
            for topic,words in topic_words:
                vocabulary.append(words)
        print("vocabulary: ", vocabulary)
        
        #calculate the number of words the new textpart should have
        num_words_per_doc = round(len(document)/100 * size_new_textpart)
        
        documents = []
        for m in range(num_docs):
            #take original document without falsely relevant words as basis
            doc = document
            #add new text part
            for n in range(num_words_per_doc):
                #sample topic according to distribution
                topic = np.random.multinomial(1, added_attributions, size=1)
                topic_idx=np.where(topic[0]==1)
                topic_idx2=topic_idx[0][0]
                            
                #use a random word from the choosen topic
                voc = random.choice(vocabulary[topic_idx2])
                doc.append(voc)
            doc = str(" ".join(doc).split(','))
            documents.append(doc)
        print("documents: ", documents)
        documents_i = vec.transform(documents)
        
        #calculate the kulback leibler divergence of the original and the new document   
        # define distributions
        topic_doc_attribution = topic_dist["topic"].to_list()
        topic_dist_new_doc = topic_doc_attribution
        for i,j in enumerate(missing_positive_topics):
            topic_dist_new_doc[j] = topic_dist_new_doc[j]+added_attributions[i]
        #calculate the new relative attributions
        sum_attributions_new_doc = sum(topic_dist_new_doc)
        topic_dist_new_doc_relative = []
        for i in topic_dist_new_doc:
            rel = i/sum_attributions_new_doc
            topic_dist_new_doc_relative.append(rel)

        q = topic_doc_attribution
        p = topic_dist_new_doc_relative
        # calculate (P || Q)
        kl_pq = sum(rel_entr(p, q))
            
        return documents_i, kl_pq


#Semantic Push approach - constructive
#create new training documents out of the relevant gs words and the associated topics
#expert knowledge is taken into account via topic distribution

#especially relevant topics get a higher weighting
def SI_create(prediction, global_id, query_inst, learner, alpha, num_docs,num_words_per_doc, gs_weight, pred=False):
    from gensim.models import LdaModel
    model = models.ldamodel.LdaModel.load(lda_path)
    document = x_data[global_id].split()
    label_true = y_data_i[global_id]
    label_pred = int("".join(map(str,prediction)))
    
    #get the topic distribution over the instance 
    #and the gold standard (expert knowledge) explanation for the true/predicted label
    if pred:
        topic_dist, gs_pos, gs_neg = local_GS_TopicLIME(label_pred, global_id)
    else:
        topic_dist, gs_pos, gs_neg = local_GS_TopicLIME(label_true, global_id)
    
    #save the topic distribution in list
    topic_doc_attribution = topic_dist["topic"].to_list()
    print("topic distribution over instance: ", topic_dist)
    print("attributions ",topic_doc_attribution) 

    
    #get the topics for each word in the document in case the topic allocation is possible under a minimum probability
    topic_list = []
    for i in document:
        word_id = id2word.token2id[i]
        topic = model.get_term_topics(word_id, minimum_probability=0.0000000000000000001)
        try:
            #choose topics with highest probability and add it to list
            topic_number = max(topic, key=itemgetter(1))[0]
            topic_list.append(topic_number)
        except:
            #in case no topics can be allocated, append number -99 to list
            topic_list.append(-99)
    print("topics in document: ", topic_list)
    

    #show explanation of true/predicted class
    if pred:
        topic_lime = TopicLIME_explainer(global_id, label_pred, query_inst, learner,alpha)
    else:
        topic_lime= TopicLIME_explainer(global_id, label_true, query_inst, learner,alpha)
    
    #transfer the topics from the topiclime explanation into list
    #divided in positiv and negative attributed topics
    positive_topics_expl = []
    negative_topics_expl = []
    for i in topic_lime:
        if i[1] >= 0:
            if str(i[0][7]).isspace():
                positive_topics_expl.append(int(i[0][8:9]))
            else:
                positive_topics_expl.append(int(i[0][7:9]))
        else:
            if str(i[0][7]).isspace():
                negative_topics_expl.append(int(i[0][8:9]))
            else:
                negative_topics_expl.append(int(i[0][7:9]))
    print("positive_topics_expl: ", positive_topics_expl)
    print("negative_topics_expl: ", negative_topics_expl)
    
    #save the top 7 positive and negative topics and their attributions from gold stadard in lists
    positive_topics = list(gs_pos.index.values)[:alpha]
    print("positive topics: ", positive_topics)
    positive_topics_gs_attribution = list(gs_pos["Gold Standard attribution"].values)[:7]
    print("positive topic gs attribution: ", positive_topics_gs_attribution)
    negative_topics = list(gs_neg.index.values)[:alpha]
    print("negative topics: ", negative_topics)
    negative_topics_gs_attribution = list(gs_neg["Gold Standard attribution"].values)[:7]
    print("negative topic gs attribution: ", negative_topics_gs_attribution)
    
    
    #turn the positive GS attribution values into relative ones
    sum_attributions_gs = sum(positive_topics_gs_attribution)
    relative_positive_topics_gs_attribution = []
    for i in positive_topics_gs_attribution:
        rel = i/sum_attributions_gs
        relative_positive_topics_gs_attribution.append(rel)
    print("relative gold standard attribution: ", relative_positive_topics_gs_attribution)
    
    #create a new weighted topic distribution accoding to the right or wrong explanations of the classifier
    lam = gs_weight
    weighted_topic_dist = []
    lda_topics= model.show_topics(num_topics=model.num_topics, num_words=40, log=False, formatted=False)
    vocabulary = []

    #for i in topic_dist.index.values:
    for i, row in topic_dist.iterrows():
        #keep the original topic distribution
        if (i in positive_topics_expl and i in positive_topics) or (i in negative_topics_expl and i in negative_topics) or (i in positive_topics_expl and i in negative_topics) or (i not in positive_topics_expl and i not in negative_topics_expl and i not in positive_topics and i not in negative_topics) or (i in negative_topics and i not in positive_topics_expl and i not in negative_topics_expl):
            weighted_topic_dist.append(topic_doc_attribution[i])
        #reinforce the original topic distribution
        if (i in negative_topics_expl and i in positive_topics) or (i in positive_topics and i not in positive_topics_expl and i not in negative_topics_expl):
            idx_gs_pos = positive_topics.index(i)
            #w.append(weight_doc[i-1]+[(lam*x) + ((1-lam)*y) for x, y in zip(weight_gs_pos[idx_gs_pos],weight_doc[i-1])])
            weighted_topic_dist.append(topic_doc_attribution[i] + lam*relative_positive_topics_gs_attribution[idx_gs_pos] + (1-lam)*topic_doc_attribution[i])
        #weaken the original topic distribution
        if (i in positive_topics_expl and i not in positive_topics and i not in negative_topics) or (i in negative_topics_expl and i not in positive_topics and i not in negative_topics):
            weighted_topic_dist.append(0)
        
        #create a vocabulary list from the lda topic words
        topic_words = [(lda_topics[i][0], [wd[0] for wd in lda_topics[i][1]])]
        for topic,words in topic_words:
            vocabulary.append(words)
    
    
    #turn the weighted topic attribution values into relative ones
    sum_weighted_topic_dist = sum(weighted_topic_dist)
    relative_weighted_topic_dist = []
    for i in weighted_topic_dist:
        rel = i/sum_weighted_topic_dist
        relative_weighted_topic_dist.append(rel)
    print("relative weighted topic distribution: ", relative_weighted_topic_dist)
    
    #create new documents according to the new weighted distribution
    documents = []
    for m in range(num_docs):
        doc = []
        for n in range(num_words_per_doc):
            #sample topic according to distribution
            topic = np.random.multinomial(1, relative_weighted_topic_dist, size=1)
            #--> list with 0s and one 1, which marks the choosen topic
            topic_idx=np.where(topic[0]==1)
            topic_idx2=topic_idx[0][0]
                        
            #use a random vocabulary word from the choosen topic
            voc = random.choice(vocabulary[topic_idx2])
            doc.append(voc)
        doc = str(" ".join(doc).split(','))
        documents.append(doc)
    #print("documents: ", documents)
    documents_i = vec.transform(documents)
    

    #Evaluation kullback leibler (divergence of the original and the new document )  
    # define distributions
    q = topic_doc_attribution
    p = relative_weighted_topic_dist
    # calculate (P || Q)
    kl_pq = sum(rel_entr(p, q))
    
    
    return documents_i, kl_pq

