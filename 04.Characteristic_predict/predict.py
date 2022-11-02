#!/usr/bin/python
# -*- coding: UTF-8 -*-

#Usage: python transferRF.py SourceData QueryData TransferData

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

def Balanced_acc(y_pred, ytest):
    acc_c = 0
    acc_class = {}
    for c in np.unique(ytest):
        i = ytest == c
        correct = y_pred[i] == ytest[i]
        acc_c += sum(correct) / len(correct)
        acc_class[c] = format(sum(correct) / len(correct), '.5f')

    Bacc = format(acc_c / len(np.unique(ytest)), '.5f')
    return Bacc, acc_class

metadata = pd.read_csv("/data4/xiongguangzhou/02.Transit/00.Data/Dataset1/metadata/metadata.csv", header=0)
for i in ("city_total_population","city_population_density","city_latitude","city_longitude","city_elevation_meters","city_ave_june_temp_c"):
    metadata[i] = list(pd.cut(metadata[i], 3, labels=['low', 'medium', 'high']))
metadata = metadata.fillna("Unknow")

SourceData, QueryData, TransferData = feature_merge(argv[1], argv[2], argv[3])

for feature in ("city","coastal_city","city_total_population","city_population_density","city_latitude","city_longitude","city_elevation_meters","city_koppen_climate","city_ave_june_temp_c"):
    
    #构建以sampleid为键和特征为值的字典
    meta_dic = convert_to_dic(metadata["sampleid"], metadata[feature])

    S_statistic, label_dic, x_source, y_source = data_process(SourceData, meta_dic, dic=True)
    Q_statistic, x_query, y_query = data_process(QueryData, meta_dic)
    T_statistic, x_transfer, y_transfer = data_process(TransferData, meta_dic)
    Nclasses = len(np.unique(y_source))
      
    #*** Train random forest classifier on source data only ***
    so_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=13)
    so_RF = so_RF.fit(x_source, y_source)
    so_y_pred = so_RF.predict(x_query)
    so_y_proba = so_RF.predict_proba(x_query)

    #*** Train random forest classifier on transfer data only ***
    to_RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=13)
    to_RF = to_RF.fit(x_transfer, y_transfer)
    to_y_pred = to_RF.predict(x_query)
    to_y_proba = to_RF.predict_proba(x_query)

    #*** Use MIX(SER + STRUT) to enhance forest trained on Source with target data ***
    ##*** SER ***
    gRF_list = trf.forest_convert(so_RF)
    ser_RF = trf.forest_SER(gRF_list, x_transfer, y_transfer, C=Nclasses)
    ser_y_pred = trf.predict(ser_RF, x_query)
    ser_y_proba = trf.predict_proba(ser_RF, x_query)

    ##*** STRUT ***
    strut_RF = trf.STRUT(x_source, y_source, x_transfer, y_transfer, n_trees=100, verbos=False)
    strut_y_pred = trf.predict(strut_RF, x_query)
    strut_y_proba = trf.predict_proba(strut_RF, x_query)

    ##*** MIX ***
    mix_y_pred = trf.mix_predict(ser_RF, strut_RF, x_query)
    mix_y_proba = trf.mix_predict_proba(ser_RF, strut_RF, x_query)
    
    #提取query sampleid
    test = pd.read_csv(argv[2])
    q_sampleid = list(test.columns)
    q_sampleid.pop(0)
            
    #构建sampleid与预测值和实际值的字典
    query_dic = {}
    so_pred_dic = {}
    to_pred_dic = {}
    ser_pred_dic = {}
    strut_pred_dic = {}
    mix_pred_dic = {}
    for i in range(len(q_sampleid)):
        query_dic[q_sampleid[i]] = y_query[i]
        so_pred_dic[q_sampleid[i]] = so_y_pred[i]
        to_pred_dic[q_sampleid[i]] = to_y_pred[i]
        ser_pred_dic[q_sampleid[i]] = ser_y_pred[i]
        strut_pred_dic[q_sampleid[i]] = strut_y_pred[i]
        mix_pred_dic[q_sampleid[i]] = mix_y_pred[i]
            
    #提取query sample的metadata
    query_meta = metadata.loc[metadata['sampleid'].isin(q_sampleid)]
            
    for city in np.unique(query_meta['city']).tolist():
            
        #提取每个城市query sampleid
        sampleid = list(query_meta.query('city == @city')['sampleid'])
            
        #提取实际值和预测值
        query = []
        so_pred = []
        to_pred = []
        ser_pred = []
        strut_pred = []
        mix_pred = []
        for id in sampleid:
            query.append(query_dic[id])
            so_pred.append(so_pred_dic[id])
            to_pred.append(to_pred_dic[id])
            ser_pred.append(ser_pred_dic[id])
            strut_pred.append(strut_pred_dic[id])
            mix_pred.append(mix_pred_dic[id])
                
        query = np.array(query)
        so_pred = np.array(so_pred)
        to_pred = np.array(to_pred)
        ser_pred = np.array(ser_pred)
        strut_pred = np.array(strut_pred)
        mix_pred = np.array(mix_pred)
            
        #计算accuracy
        so_acc, so_acc_class = Balanced_acc(so_pred, query)
        to_acc, to_acc_class = Balanced_acc(to_pred, query)
        ser_acc, ser_acc_class = Balanced_acc(ser_pred, query)
        strut_acc, strut_acc_class = Balanced_acc(strut_pred, query)
        mix_acc, mix_acc_class = Balanced_acc(mix_pred, query)

        print(city, feature, "base", so_acc_class, sep=",")
        print(city, feature, "independent", to_acc_class, sep=",")
        print(city, feature, "transfer", mix_acc_class, sep=",")
