# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:58:48 2021

@author: Mareike Hoffmann
"""
#README: Implementation of explainer functions LIME & TopicLIME
#The TopicLIME approach is by Sebastian M. Kiefer (née Bruckert) & Jonas R. Amling


from gensim.models import LdaModel
import gensim.models
import lime
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from lime_topic_final.lime_text_topics import LimeTextByTopicsExplainer

from init import *
from lda import id2word

    
#LIME Explanation
def LIME_explainer(global_id, label, query_inst, learner, alpha):  
    # prediction = learner.predict(query_inst)
    # true_label = y_data_i[global_id]
    #class names expand dynamically the more labels the learner know
    class_names_labeled = np.unique(learner.y_training)
    #LIME uses indexes as class names; therefore the assignment to the real class name is needed
    # if expl_for_true_label == False:
    #     expl_idx=class_names_labeled.tolist().index(int(prediction)) 
    # else:
    #     expl_idx=class_names_labeled.tolist().index(int(true_label))
    # if expl_for_true_label == False:
    #     label=int(prediction)
    # else:
    #     label=int(true_label)
    
    c = make_pipeline(vec, learner)

    #change list format to strings
    text_instance = str(x_data[global_id][1:-1])
    
    explainer = LimeTextExplainer(class_names=class_names_labeled)
    # exp = explainer.explain_instance(text_instance, c.predict_proba, top_labels=len(class_names_labeled), num_features=alpha)
    exp = explainer.explain_instance(text_instance, c.predict_proba, num_features=alpha, labels=[label])
    
    # print('\n\n********************* Standard LIME ***********************\n\n')
    # for x in exp.available_labels():
    #     if x == expl_idx:
    #         lime_exp = exp.as_list(label=x)
    #         print('Word-based explanation for class %s' % class_names_labeled[x])
    #         print('\n'.join(map(str, lime_exp)))
    #         print()
    print('\n\n********************* Standard LIME ***********************\n\n')
    for x in exp.available_labels():
        lime_exp = exp.as_list(label=x)
        print('Word-based explanation for class %s' % class_names_labeled[x])
        print('\n'.join(map(str, lime_exp)))
        print()

    return lime_exp

#LIME Explainer for Evaluation
def LIME_explainer_eval(instance, true_class, learner, len_explanation, test=False):
    if test:
        class_names_labeled = np.unique(y_test_i)
    else:
        class_names_labeled = np.unique(learner.y_training)

    #expl_idx=class_names_labeled.tolist().index(int(true_class))
    label=int(true_class)

    #change list format to strings
    if test:
        text_instance = str(instance[1:-1])
    else:
        text_instance = str(x_data[instance][1:-1])
    
    c = make_pipeline(vec, learner)
    #print("probabilities ", c.predict_proba(instance))
    explainer = LimeTextExplainer(class_names=class_names_labeled)
    #exp = explainer.explain_instance(text_instance, c.predict_proba, top_labels=len(class_names_labeled), num_features=len_explanation)
    #print("label: " + str([label]))
    exp = explainer.explain_instance(text_instance, c.predict_proba, num_features=len_explanation, labels=[label])
    
    # for x in exp.available_labels():
    #     if x == expl_idx:
    #         lime_exp = exp.as_list(label=x)
    for x in exp.available_labels():
        lime_exp = exp.as_list(label=x)
    
    #transfer the words from the lime explanation into list
    lime_exp_list = []
    for i in lime_exp:
        lime_exp_list.append(i[0])

    return lime_exp_list
    

#Function to use TopicLIME
def word_to_topics(word):
    """
    Maps a word on its corresponding topics
    :param word: the word that is searched for
    :return: list of topics
    """
    model = LdaModel.load(lda_path)
    
    if word in id2word.token2id:
            word_id = id2word.token2id[word]
            # so im paper ecml verwernet: if lda_model.get_term_topics(word_id, minimum_probability=0.000000000000001):
            if model.get_term_topics(word_id, minimum_probability=0.0000000000000000001):
                    #so in ECML verwendet z = [x for x in lda_model.get_term_topics(word_id, minimum_probability=0.0000000000000000001)]
                    z = [x for x in model.get_term_topics(word_id, minimum_probability=0.0000000000000000001)]
                    z_sorted = sorted(z, key=lambda tupel:tupel[1], reverse=True)
                    #print(str([z[0] for z in z_sorted[:1]]))
                    return [z[0] for z in z_sorted[:1]]

            else:
                #so in ECML verwerndet z = [x for x in lda_model.get_term_topics(word_id, minimum_probability=0.0000000000000000003)]
                z = [x for x in model.get_term_topics(word_id, minimum_probability=0.0000000000000000003)]
                z_sorted = sorted(z, key=lambda tupel: tupel[1], reverse=True)
                #print(str([z[0] for z in z_sorted[:1]]))
                return [z[0] for z in z_sorted[:1]]
    else:
        print("word " + word + " is out of vocabulary")
        return[]


#TopicLIME Explanation
def TopicLIME_explainer(global_id, label, query_inst, learner, len_explanation):
    # prediction = learner.predict(query_inst)
    # true_label = y_data_i[global_id]
    #class names expand dynamically the more labels the learner know
    class_names_labeled = np.unique(learner.y_training)
    # if expl_for_true_label == False:
    #     expl_idx=class_names_labeled.tolist().index(int(prediction)) 
    # else:
    #     expl_idx=class_names_labeled.tolist().index(int(true_label))
    # if expl_for_true_label == False:
    #     label=int(prediction)
    # else:
    #     label=int(true_label)
    
    c = make_pipeline(vec, learner)

    #change list format to strings
    text_instance = str(x_data[global_id])[1:-1] 
    
    model = LdaModel.load(lda_path)
    
    topics = []
    for x in range(model.num_topics):
        if x <10:
            topics.append('topic # ' + str(x))
        else:
            topics.append('topic #' + str(x))
    
    explainer_mod = LimeTextByTopicsExplainer(class_names=class_names_labeled, consider_all_words=False, word_to_topics=word_to_topics, topics=topics, feature_selection="forward_selection", random_state=54321, verbose=False)
    #exp = explainer_mod.explain_instance(text_instance, c.predict_proba, num_features=len_explanation, top_labels=len(class_names_labeled))
    exp = explainer_mod.explain_instance(text_instance, c.predict_proba, num_features=len_explanation, labels=[label])
    
    # print('\n\n********************* topicLIME ***********************\n\n')
    # for x in exp.available_labels():
    #     if x == expl_idx:
    #         topiclime = exp.as_list(label=x)
    #         print('Topic-based explanation for class %s' % class_names_labeled[x])
    #         print('\n'.join(map(str, topiclime)))
    #         print()
    print('\n\n********************* topicLIME ***********************\n\n')
    for x in exp.available_labels():
        topiclime = exp.as_list(label=x)
        print('Topic-based explanation for class %s' % class_names_labeled[x])
        print('\n'.join(map(str, topiclime)))
        print()
    
    return topiclime


#topicLIME Explainer for Evaluation
def topicLIME_explainer_eval(instance, true_class, learner, len_explanation, test=False):
    if test:
        class_names_labeled = np.unique(y_test_i)
    else:
        class_names_labeled = np.unique(learner.y_training)
    #expl_idx=class_names_labeled.tolist().index(int(true_class))
    label = int(true_class)
    
    #change list format to strings
    if test:
        text_instance = str(instance[1:-1])
    else:
        text_instance = str(x_data[instance][1:-1])
    
    c = make_pipeline(vec, learner)
    model = LdaModel.load(lda_path)
    
    topics = []
    for x in range(model.num_topics):
        if x <10:
            topics.append('topic # ' + str(x))
        else:
            topics.append('topic #' + str(x))
    
    explainer_mod = LimeTextByTopicsExplainer(class_names=class_names_labeled, consider_all_words=False, word_to_topics=word_to_topics, topics=topics, feature_selection="forward_selection", random_state=54321, verbose=False)
    #exp = explainer_mod.explain_instance(text_instance, c.predict_proba, num_features=len_explanation, top_labels=len(class_names_labeled))
    exp = explainer_mod.explain_instance(text_instance, c.predict_proba, num_features=len_explanation, labels=[label])
    
    # for x in exp.available_labels():
    #     if x == expl_idx:
    #         print("label : ", x)
    #         topiclime = exp.as_list(label=x)
    for x in exp.available_labels():
        print("label : ", x)
        topiclime = exp.as_list(label=x)
    
    #transfer the words from the topiclime explanation into list
    tl_exp_list = []
    for i in topiclime:
        if str(i[0][7]).isspace():
            tl_exp_list.append(int(i[0][8:9]))
        else:
            tl_exp_list.append(int(i[0][7:9]))
    print("tl topics ",tl_exp_list)
        
    return tl_exp_list