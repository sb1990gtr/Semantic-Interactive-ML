
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from xgboost import XGBClassifier
from statistics import mean
from init import *
import pickle

#LIME imports
from semanticInterrogations import LIME_SI_const, LIME_SI_dest, SI_create
from explainer import LIME_explainer_eval
from goldStandard import local_GS_eval

#topicLIME / SI imports
from semanticInterrogations import TopicLIME_SI_const, TopicLIME_SI_dest
from semanticInterrogations import SI_correct
from explainer import topicLIME_explainer_eval
from goldStandard import local_GS_TopicLIME_eval


#initializing the learner
learner = ActiveLearner(
    estimator=XGBClassifier(),
    query_strategy=uncertainty_sampling,
    X_training=x_labeled, y_training=y_labeled
)


#Parameters for S2)-S4)
#determine the number of new training documents
num_docs = 10
#length of each document is equal to the average document length
num_words_per_doc = 25
#number of words of LIME gold standard pos & neg each and LIME explanation
alpha_LIME = 7
#number of topics of topicLIME gold standard pos & neg each and TL explanation
alpha_TL = 3
#percentage that specifies how many words from GS shall be taken as relevant for words and topics respectively, relevant for RQ2
percentage=0.1
percentage_TL=0.1

#Parameters only relevant for S4)
#determine the size (%) the new textpart should have in proportion to the original document
size_new_textpart = 50
#determine the weight the gs topic attribution should have compared to the attribution of topics in document
gs_weight = 0.95
#number of topics of topicLIME gold standard pos & neg each and SI explanation
alpha_SI_create = 3

right=[]
#Lists for Evaluation
#Kullback Leibler of new training document vs. original documentd
kullback_leibler_pq_dest =[]
kullback_leibler_pq_const =[]
kullback_leibler_pq_counterexample=[]

#RQ1: Performance Trainset
weighted_avg_trainset=[]
macro_avg_trainset = []
accuracy_trainset = []
#initial Performance Trainset
report_trainset = metrics.classification_report(learner.y_training, learner.predict(learner.X_training), output_dict=True)
weighted_avg_trainset.append(report_trainset["weighted avg"]["f1-score"])
macro_avg_trainset.append(report_trainset["macro avg"]["f1-score"])
accuracy_trainset.append(report_trainset["accuracy"])

#RQ1: Performance Testset
weighted_avg_testset=[]
macro_avg_testset = []
accuracy_testset = []
#initial Performance Testset
report_testset = metrics.classification_report(y_test_i, learner.predict(x_test_i), output_dict=True)
weighted_avg_testset.append(report_testset["weighted avg"]["f1-score"])
macro_avg_testset.append(report_testset["macro avg"]["f1-score"])
accuracy_testset.append(report_testset["accuracy"])

#RQ1: number of correct predictions (TP)
correct_predictions_testset = []
correct_predictions_trainset = []

#RQ1: Classification margin for re-classified queries/test instances
x_queries = []
y_queries = [] 
globalid_queries=[]
confidence_diff_pred_true_avg = []
confidence_diff_pred_true_avg_test=[]

#RQ2: Explanation Quality Queries 
correct_explanations_queries_LIME = []
correct_explanations_avg_queries_LIME =[]
correct_explanations_queries_LIME_labeled = []
correct_explanations_avg_queries_LIME_labeled =[]
correct_explanations_queries_topicLIME = []
correct_explanations_avg_queries_topicLIME =[]

#RQ2: Explanation Quality Testset
correct_explanations_avg_testset_LIME=[]
correct_explanations_avg_testset_topicLIME=[]
#value at starting point
# correct_words_relative = []
# for i,j in enumerate(x_test):
#     print(i)
#     true_class = y_test_i[i]
#     #test predict proba
#     gs_true_class = local_GS_eval(true_class, j, test=True, percentage=percentage)
#     lime_true_class = LIME_explainer_eval(j, true_class, learner, len(gs_true_class), test=True)
#     correct_words = set(lime_true_class)&set(gs_true_class)
#     correct_words_relative.append(len(correct_words)/len(gs_true_class))
# correct_explanations_avg_testset_LIME.append(mean(correct_words_relative))
#correct_topics_relative = []
#for i,j in enumerate(x_test):
#         true_class = y_test_i[i]
#         gs_true_class = local_GS_TopicLIME_eval(true_class, j, test=True, percentage_TL=percentage_TL)
#         topicLIME_true_class = topicLIME_explainer_eval(j, true_class, learner, len(gs_true_class), test=True)
#         correct_topics = set(topicLIME_true_class)&set(gs_true_class)
#         correct_topics_relative.append(len(correct_topics)/len(gs_true_class))
#     correct_explanations_avg_testset_topicLIME.append(mean(correct_topics_relative))


#####################Start of active learning process##############
iteration = 1

while iteration < 200:
    print("Iteration: ", iteration)
       
    #query for labels
    query_idx, query_inst = learner.query(x_pool)
    query_idx_int = int(query_idx)
    global_id = indices_pool[query_idx_int]
    prediction =  learner.predict(query_inst)
    
    #RQ1: save selected queries and true class
    #x_queries.append(vec.transform([x_data[global_id]]))
    x_queries.append(query_inst)
    y_queries.append(y_data_i[global_id])
    globalid_queries.append(global_id)
    
    print("Doument id: ", global_id)
    print("Predicted class: ", prediction, le.inverse_transform(prediction))
    print("True class: ", y_data_i[global_id], y_data[global_id])
    
    # S1)
    # Base Case

    learner.teach(x_pool[query_idx], y_pool[query_idx])

    
    #S2)
    #LIME
    #if prediction is true, correct the explanation if necessary - "right for the wrong reasons"
    #if prediction is false, correct the prediction - "provide the correct label" and constructive correction
    if prediction == [y_data_i[global_id]]:
        #correct the original instance
        learner.teach(x_pool[query_idx], y_pool[query_idx])
        x_correction_i = LIME_SI_dest(global_id, prediction, query_inst, learner, alpha_LIME, num_docs)
        y_correction_i = [y_data_i[global_id]] * num_docs
        learner.teach(x_correction_i, y_correction_i)
        right.append(1)
    else:
        #correct the original instance
        learner.teach(x_pool[query_idx], y_pool[query_idx])
        x_correction_i = LIME_SI_const(global_id, query_inst, learner, alpha_LIME, num_docs, num_words_per_doc)
        y_correction_i = [y_data_i[global_id]] * num_docs
        learner.teach(x_correction_i, y_correction_i)
        right.append(-99)


    
    # #S4) Semantic Interrogations
    if prediction == [y_data_i[global_id]]:
        learner.teach(x_pool[query_idx], y_pool[query_idx])
        if gs_model_topic.predict(vec) == [y_data_i[global_id]]:
            x_correction_i, kl_pq = SI_correct(global_id, prediction, query_inst, learner, alpha_TL, num_docs, size_new_textpart, gs_weight)
            y_correction_i = [y_data_i[global_id]] * num_docs
            learner.teach(x_correction_i, y_correction_i)
            kullback_leibler_pq_dest.append(kl_pq)
    else:
        #teach with the original document and the correct label
        learner.teach(x_pool[query_idx], y_pool[query_idx])
        #create new documents based on the corrections of the wrong explanations
        x_correction_i, kl_pq = SI_create(prediction,global_id, query_inst, learner, alpha_SI_create, num_docs,num_words_per_doc, gs_weight, pred=False)
        y_correction_i = [y_data_i[global_id]] * num_docs
        learner.teach(x_correction_i, y_correction_i)
        kullback_leibler_pq_const.append(kl_pq)

        #train for the wrong predicted label via counterexamples
        x_correction_j, kl_pq_ce = SI_create(prediction,global_id, query_inst, learner, alpha_SI_create, num_docs,num_words_per_doc, gs_weight, pred=True)
        y_correction_j = list(prediction) * num_docs
        learner.teach(x_correction_j, y_correction_j)
        kullback_leibler_pq_counterexample.append(kl_pq_ce)

    
        
    ######################### Evaluation ###################################
    #RQ1 Testset Evaluation
    report_testset = metrics.classification_report(y_test_i, learner.predict(x_test_i), output_dict=True)
    macro_avg_testset.append(report_testset["macro avg"]["f1-score"])
    print("RQ1 macro f1s_testset ", macro_avg_testset)

    RQ1: Classification margin for re-classified queries
    if iteration%10 ==0:
        confidence_diff_pred_true = []
        class_names_known = np.unique(learner.y_training)
        for i, j in enumerate(x_queries):
            pred = learner.predict_proba(j)
            pred = np.squeeze(pred.tolist())
            idx_true_class_in_pred_list = int(np.where(class_names_known==y_queries[i])[0])
            difference_predicted_true = max(pred) - pred[idx_true_class_in_pred_list]
            #difference_predicted_true = max(pred) - pred[y_queries[i]]
            confidence_diff_pred_true.append(difference_predicted_true)
        confidence_diff_pred_true_avg.append(mean(confidence_diff_pred_true))
    print("RQ1 avg difference of confidence between predicted and true class on selected queries ", confidence_diff_pred_true_avg)
    
    #RQ1: Classification margin for classified test instances
    if iteration%10 ==0:
        confidence_diff_pred_true = []
        class_names_known = np.unique(y_test_i)
        for i, j in enumerate(x_test_i):
            pred = learner.predict_proba(j)
            pred = np.squeeze(pred.tolist())
            idx_true_class_in_pred_list = int(np.where(class_names_known==y_test_i[i])[0])
            difference_predicted_true = max(pred) - pred[idx_true_class_in_pred_list]
            #difference_predicted_true = max(pred) - pred[y_test_i[i]]
            confidence_diff_pred_true.append(difference_predicted_true)
        confidence_diff_pred_true_avg_test.append(mean(confidence_diff_pred_true))
    print("RQ1 avg difference of confidence between predicted and true class on testset ", confidence_diff_pred_true_avg_test)
    
    #
    # #RQ2 Explanation Quality Queries
    # #LIME
    for i,j in enumerate(globalid_queries):
        true_class = y_queries[i]
        gs_true_class = local_GS_eval(true_class, j, test=False, percentage=percentage)
        lime_true_class = LIME_explainer_eval(globalid_queries[i], true_class, learner, len(gs_true_class), test=False)
        correct_words = set(lime_true_class)&set(gs_true_class)
        correct_explanations_queries_LIME.append(len(correct_words)/len(gs_true_class))
    correct_explanations_avg_queries_LIME.append(mean(correct_explanations_queries_LIME))
    #print("RQ2 number of correct words/topics in explanation of queries LIME: ", correct_explanations_queries_LIME)
    print("RQ2 average number of correct words in explanation of queries LIME: ", correct_explanations_avg_queries_LIME)
    for i,j in enumerate(x_labeled):
        global_id = indices_labeled[i]
        true_class = y_labeled[i]
        #print("global_id_error: " + str(global_id))
        gs_true_class = local_GS_eval(true_class, global_id, test=False, percentage=percentage)
        lime_true_class = LIME_explainer_eval(global_id, true_class, learner, len(gs_true_class), test=False)
        correct_words = set(lime_true_class)&set(gs_true_class)
        correct_explanations_queries_LIME_labeled.append(len(correct_words)/len(gs_true_class))
    correct_explanations_avg_queries_LIME_labeled.append(mean(correct_explanations_queries_LIME_labeled))
    print("RQ2 number of correct words/topics in explanation of queries LIME: ", correct_explanations_queries_LIME)
    print("RQ2 average number of correct words in explanation of labeled LIME: ", correct_explanations_avg_queries_LIME_labeled)

    topicLIME / SI
    for i,j in enumerate(globalid_queries):
        true_class = y_queries[i]
        gs_true_class = local_GS_TopicLIME_eval(true_class, j, test=False, percentage_TL=percentage_TL)
        topicLIME_true_class = topicLIME_explainer_eval(globalid_queries[i], true_class, learner, len(gs_true_class), test=False)
        correct_topics = set(topicLIME_true_class)&set(gs_true_class)
        correct_explanations_queries_topicLIME.append(len(correct_topics)/len(gs_true_class))
    correct_explanations_avg_queries_topicLIME.append(mean(correct_explanations_queries_topicLIME))
    print("RQ2 number of correct words/topics in explanation of queries topicLIME: ", correct_explanations_queries_topicLIME)
    print("RQ2 average number of correct words/topics in explanation of queries topicLIME: ", correct_explanations_avg_queries_topicLIME)

    
    RQ2 Explanation Quality Testset
    word based
    if iteration%20 ==0:
        correct_words_relative = []
        for i,j in enumerate(x_test):
            #print(i)
            true_class = y_test_i[i]
            #test predict proba
            gs_true_class = local_GS_eval(true_class, j, test=True, percentage=percentage)
            lime_true_class = LIME_explainer_eval(j, true_class, learner, len(gs_true_class), test=True)
            correct_words = set(lime_true_class)&set(gs_true_class)
            correct_words_relative.append(len(correct_words)/len(gs_true_class))
        correct_explanations_avg_testset_LIME.append(mean(correct_words_relative))
    print("RQ2 overall explanation qality on testset LIME: ", correct_explanations_avg_testset_LIME)

    topicLIME / SI
    if iteration%20 ==0:
        correct_topics_relative = []
        for i,j in enumerate(x_test):
            true_class = y_test_i[i]
            gs_true_class = local_GS_TopicLIME_eval(true_class, j, test=True, percentage_TL=percentage_TL)
            topicLIME_true_class = topicLIME_explainer_eval(j, true_class, learner, len(gs_true_class), test=True)
            correct_topics = set(topicLIME_true_class)&set(gs_true_class)
            correct_topics_relative.append(len(correct_topics)/len(gs_true_class))
        correct_explanations_avg_testset_topicLIME.append(mean(correct_topics_relative))
    print("RQ2 overall explanation qality on testset topicLIME: ", correct_explanations_avg_testset_topicLIME)
        
    
    iteration += 1
learner.estimator.save_model(main_path + '/Learner/LearnerS1_bis200.model')