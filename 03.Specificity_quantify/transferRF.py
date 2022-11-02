#!/usr/bin/python
# -*- coding: UTF-8 -*-

#Usage: python transferRF.py SourceData TransferData QueryData BatchNum

from __future__ import division
import threading

from pandas.core.arrays.sparse import dtype
import numpy as np
import pandas as pd
from joblib import Parallel
from sklearn.utils.fixes import delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from sys import argv

import sys
sys.path.append('/data4/xiongguangzhou/02.Transit/TransferRandomForest/')
import TransferRandomForest as trf

def label_convert(labels, dic = False):
    Nclasses = np.unique(labels).tolist()
    labeldic = {}
    #labels = labels.tolist()
    for i in Nclasses:
        labeldic[i] = Nclasses.index(i)
    for i in range(len(labels)):
        labels[i] = labeldic[labels[i]]
    if dic == True:
        return np.array(labels), labeldic
    else:
        return np.array(labels)     

def convert_to_dic(narray_key, narray_value):
    list_key = narray_key.tolist()
    list_value = narray_value.tolist()

    convert_dic = {}
    for i in range(len(list_key)):
        convert_dic[list_key[i]] = list_value[i]

    return convert_dic

def feature_merge(Sourefile, Queryfile, Transferfile):
    SourceData = pd.read_csv(Sourefile, index_col=None, verbose=False)
    QueryData = pd.read_csv(Queryfile, index_col=None, verbose=False)
    TransferData = pd.read_csv(Transferfile, index_col=None, verbose=False)

    s_feature = set(SourceData["tax"])
    q_feature = set(QueryData["tax"])
    t_feature = set(TransferData["tax"])

    feature = set.union(s_feature, q_feature, t_feature)

    sn_feature = np.array(list(feature.symmetric_difference(s_feature)))
    qn_feature = np.array(list(feature.symmetric_difference(q_feature)))
    tn_feature = np.array(list(feature.symmetric_difference(t_feature)))

    tmpdata = np.zeros(shape=(len(sn_feature), len(SourceData.columns)-1))
    tmpdata = pd.DataFrame(np.c_[sn_feature, tmpdata], columns=SourceData.columns)
    SourceData = pd.concat([SourceData, tmpdata])

    tmpdata = np.zeros(shape=(len(qn_feature), len(QueryData.columns)-1))
    tmpdata = pd.DataFrame(np.c_[qn_feature, tmpdata], columns=QueryData.columns)
    QueryData = pd.concat([QueryData, tmpdata])

    tmpdata = np.zeros(shape=(len(tn_feature), len(TransferData.columns)-1))
    tmpdata = pd.DataFrame(np.c_[tn_feature, tmpdata], columns=TransferData.columns)
    TransferData = pd.concat([TransferData, tmpdata])
    return SourceData, QueryData, TransferData

def data_process(tax_table, metadata_dic, dic=False):
    data = tax_table.T
    data.columns = data.loc["tax"]
    data.drop(["tax"], inplace=True)

    features_table = data.to_numpy(dtype=np.float_)
    sampleid = list(data.index)
    labels = []
    for i in range(len(sampleid)):
        labels.append(metadata_dic[sampleid[i]])
    #labels = labels.to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    statistic = convert_to_dic(classes, counts)
    labels, labels_dic = label_convert(labels, dic=True)
    if dic == True:
        return statistic, labels_dic, features_table, labels
    else:
        return statistic, features_table, labels

def Balanced_acc(y_pred, ytest, Nclass):
    acc_c = 0
    acc_class = {}
    for c in np.unique(ytest):
        i = ytest == c
        correct = y_pred[i] == ytest[i]
        acc_c += sum(correct) / len(correct)
        acc_class[c] = format(sum(correct) / len(correct), '.5f')

    Bacc = format(acc_c / len(np.unique(ytest)), '.5f')
    return Bacc, acc_class

def roc_auc_calculate(y_query, y_proba):
    classes = np.unique(y_query)
    Nclasses = len(classes)
    y_test = np.zeros((len(y_query), Nclasses))
    for i in range(len(y_query)):
        y_test[i][y_query[i]] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(Nclasses):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba[:, i])
        roc_auc[i] = float(format(auc(fpr[i], tpr[i]), '.5f'))
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(Nclasses)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(Nclasses):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= Nclasses

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = float(format(auc(fpr["macro"], tpr["macro"]), '.5f'))
    

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_proba.ravel())
    roc_auc["micro"] = float(format(auc(fpr["micro"], tpr["micro"]), '.5f'))
    
    roc_auc["ovr"] = float(format(roc_auc_score(y_query, y_proba, multi_class='ovr'), '.5f'))
    roc_auc["ovo"] = float(format(roc_auc_score(y_query, y_proba, multi_class='ovo'), '.5f'))
    return roc_auc, fpr, tpr

def roc_curve_plot(roc_auc, fpr, tpr, filename):
    plt.figure()
    lw = 2
    
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "deeppink", "navy"])
    for i, color in zip(range(Nclasses), colors):
        plt.plot( fpr[i], tpr[i], color=color, lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),)

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Transfer Random Forest ROC curve")
    plt.legend(loc="lower right")
    plt.show()
    filepath = "./" + filename + ".pdf"
    plt.savefig(filepath)

metadata = pd.read_csv("/data4/xiongguangzhou/02.Transit/00.Data/Dataset1/metadata/metadata.csv", header=0)
meta_dic = convert_to_dic(metadata["sampleid"], metadata["city"])

SourceData, QueryData, TransferData = feature_merge(argv[1], argv[2], argv[3])

S_statistic, label_dic, x_source, y_source = data_process(SourceData, meta_dic, dic=True)
Q_statistic, x_query, y_query = data_process(QueryData, meta_dic)
T_statistic, x_transfer, y_transfer = data_process(TransferData, meta_dic)
Nclasses = len(np.unique(y_source))


#*** Train random forest classifier on source data only ***
so_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=13)
so_RF = so_RF.fit(x_source, y_source)
so_y_pred = so_RF.predict(x_query)
so_y_proba = so_RF.predict_proba(x_query)

so_acc, so_acc_class = Balanced_acc(so_y_pred, y_query, Nclasses)
so_roc_auc, so_fpr, so_tpr = roc_auc_calculate(y_query, so_y_proba)

#print('\nmean Acc - Source = {:.2f}'.format(so_acc), so_acc_class, '\n', so_roc_auc)
so_filename = "Source-" + argv[4]
roc_curve_plot(so_roc_auc, so_fpr, so_tpr, so_filename)


#*** Train random forest classifier on transfer data only ***
to_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=13)
to_RF = to_RF.fit(x_transfer, y_transfer)
to_y_pred = to_RF.predict(x_query)
to_y_proba = to_RF.predict_proba(x_query)

to_acc, to_acc_class = Balanced_acc(to_y_pred, y_query, Nclasses)
to_roc_auc, to_fpr, to_tpr = roc_auc_calculate(y_query, to_y_proba)

#print('mean Acc - Target = {:.2f}'.format(to_acc), to_acc_class, '\n', to_roc_auc)
to_filename = "Transfer-" + argv[4]
roc_curve_plot(to_roc_auc, to_fpr, to_tpr, to_filename)


#*** Use MIX(SER + STRUT) to enhance forest trained on Source with target data ***
##*** SER ***
gRF_list = trf.forest_convert(so_RF)
ser_RF = trf.forest_SER(gRF_list, x_transfer, y_transfer, C=Nclasses)
ser_y_pred = trf.predict(ser_RF, x_query)
ser_y_proba = trf.predict_proba(ser_RF, x_query)

ser_acc, ser_acc_class = Balanced_acc(ser_y_pred, y_query, Nclasses)
ser_roc_auc, ser_fpr, ser_tpr = roc_auc_calculate(y_query, ser_y_proba)

#print('mean Acc - SER = {:.2f}'.format(ser_acc), ser_acc_class, '\n', ser_roc_auc)
ser_filename = "SER-" + argv[4]
roc_curve_plot(ser_roc_auc, ser_fpr, ser_tpr, ser_filename)


##*** STRUT ***
strut_RF = trf.STRUT(x_source, y_source, x_transfer, y_transfer, n_trees=100, verbos=False)
strut_y_pred = trf.predict(strut_RF, x_query)
strut_y_proba = trf.predict_proba(strut_RF, x_query)

strut_acc, strut_acc_class = Balanced_acc(strut_y_pred, y_query, Nclasses)
strut_roc_auc, strut_fpr, strut_tpr = roc_auc_calculate(y_query, strut_y_proba)

#print('mean Acc - STRUT = {:.2f}'.format(strut_acc), strut_acc_class, '\n', strut_roc_auc)
strut_filename = "STRUT-" + argv[4]
roc_curve_plot(strut_roc_auc, strut_fpr, strut_tpr, strut_filename)


##*** MIX ***
mix_y_pred = trf.mix_predict(ser_RF, strut_RF, x_query)
mix_y_proba = trf.mix_predict_proba(ser_RF, strut_RF, x_query)

mix_acc, mix_acc_class = Balanced_acc(mix_y_pred, y_query, Nclasses)
mix_roc_auc, mix_fpr, mix_tpr = roc_auc_calculate(y_query, mix_y_proba)

#print('mean Acc - MIX = {:.2f}'.format(mix_acc), mix_acc_class, '\n', mix_roc_auc)
mix_filename = "MIX-" + argv[4]
roc_curve_plot(mix_roc_auc, mix_fpr, mix_tpr, mix_filename)


#** Print results ***
print("*** Data statisdic ***\nSource:   ", S_statistic, 
    "\nQuery:    ", Q_statistic, 
    "\nTransfer: ", T_statistic, "\n", 
    sep="")

print("*** Class dic ***\n", label_dic, "\n", sep="")

print("*** Accuracy ***\nSource:   ", so_acc_class, 
    "\nTransfer: ", to_acc_class, 
    "\nSER:      ", ser_acc_class, 
    "\nSTRUT:    ", strut_acc_class, 
    "\nMIX:      ", mix_acc_class, "\n", 
    sep="")
print("*** AUROC ***\nSource:   ", so_roc_auc, 
    "\nTransfer: ", to_roc_auc, 
    "\nSER:      ", ser_roc_auc, 
    "\nSTRUT:    ", strut_roc_auc, 
    "\nMIX:      ", mix_roc_auc, 
    sep="")

lw = 2
plt.figure()
plt.plot(so_fpr["micro"], so_tpr["micro"], color="deeppink", linestyle=":", linewidth=4,
    label="SO micro-average ROC curve (area = {0:0.2f})".format(so_roc_auc["micro"]),)
plt.plot(to_fpr["micro"], to_tpr["micro"], color="navy", linestyle=":", linewidth=4,
    label="TO micro-average ROC curve (area = {0:0.2f})".format(to_roc_auc["micro"]),)
plt.plot(ser_fpr["micro"], ser_tpr["micro"], color="aqua", linestyle=":", linewidth=4,
    label="SER micro-average ROC curve (area = {0:0.2f})".format(ser_roc_auc["micro"]),)
plt.plot(strut_fpr["micro"], strut_tpr["micro"], color="darkorange", linestyle=":", linewidth=4,
    label="STRUT micro-average ROC curve (area = {0:0.2f})".format(strut_roc_auc["micro"]),)
plt.plot(mix_fpr["micro"], mix_tpr["micro"], color="cornflowerblue", linestyle=":", linewidth=4,
    label="MIX micro-average ROC curve (area = {0:0.2f})".format(mix_roc_auc["micro"]),)

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Transfer Random Forest ROC curve - Micro")
plt.legend(loc="lower right")
plt.show()
micro_filename = "Micro-" + argv[4]
filepath = "./" + micro_filename + ".pdf"
plt.savefig(filepath)

plt.figure()
plt.plot(so_fpr["macro"], so_tpr["macro"], color="deeppink", linestyle=":", linewidth=4,
    label="SO macro-average ROC curve (area = {0:0.2f})".format(so_roc_auc["macro"]),)
plt.plot(to_fpr["macro"], to_tpr["macro"], color="navy", linestyle=":", linewidth=4,
    label="TO macro-average ROC curve (area = {0:0.2f})".format(to_roc_auc["macro"]),)
plt.plot(ser_fpr["macro"], ser_tpr["macro"], color="aqua", linestyle=":", linewidth=4,
    label="SER macro-average ROC curve (area = {0:0.2f})".format(ser_roc_auc["macro"]),)
plt.plot(strut_fpr["macro"], strut_tpr["macro"], color="darkorange", linestyle=":", linewidth=4,
    label="STURT macro-average ROC curve (area = {0:0.2f})".format(strut_roc_auc["macro"]),)
plt.plot(mix_fpr["macro"], mix_tpr["macro"], color="cornflowerblue", linestyle=":", linewidth=4,
    label="MIX macro-average ROC curve (area = {0:0.2f})".format(mix_roc_auc["macro"]),)

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Transfer Random Forest ROC curve - Macro")
plt.legend(loc="lower right")
plt.show()
macro_filename = "Macro-" + argv[4]
filepath = "./" + macro_filename + ".pdf"
plt.savefig(filepath)
