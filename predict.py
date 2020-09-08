
 #-*-coding:utf-8 -*-
import sys
import wordninja
import enchant
import re
import json
import random
from sklearn.cluster import KMeans
import numpy as np
import dns.resolver
import dns.resolver
import pandas as pd

# import fit.py 
import fit
 
# get center from train data
def open_train_data():
    # read fitted train data
    train_data_file = 'cluster3groups+.json'
    with open(train_data_file) as json_file:
        train_data = json.load(json_file)
    return train_data

def create_train_df(train_data, i):
    df = pd.DataFrame(columns=['creation_date', 'update_date', 'expire_date', 'dic_score', 'mx_score'])
    for index, item in enumerate(train_data[str(i)]):
        df.loc[index] = item
    df['cluster_id'] = i
    return df
        
def get_center():
    train_data = open_train_data()

    center_records = {}
    for i in range(3):
        center_records[i] = (sum(np.array(train_data[str(i)])) / len(train_data[str(i)])).tolist()
    return center_records

# calculation functions for test data 
def distance_to_center(data_point, center_point, lamda=2):
    distance = 0.0
    lamda = float(lamda)

    for a, b in zip(data_point, center_point):
        distance += (abs(a - b)**lamda)

    return distance**(1.0/lamda)

def find_closest_center(data_list, center_list):
    cluster_result = {}

    for i, data_point in enumerate(data_list):
        # define these variables: 
        # one is a null value and another is the maximum value a variable. 
        center_index = None
        min_distance = sys.maxsize

        # find a center with minimal distance
        for j, center_point in enumerate(center_list):

            distance = distance_to_center(data_point, center_point)

            if distance < min_distance:
                center_index = j
                min_distance = distance

        # assign data-point to the cluster
        cluster_result[i] = center_index

    return cluster_result

def classifying_data(data_list=None, center_list=None):
    return find_closest_center(data_list, center_list)


def test_result(test_data):
    # read train data
    train_data = open_train_data()

    # cluster list
    cluster_list = []
    for i in list(train_data.keys()):
        cluster_list.append(i)

    df_0 = create_train_df(train_data, 0)
    df_1 = create_train_df(train_data, 1)
    df_2 = create_train_df(train_data, 2)
    df_train = pd.concat([df_0, df_1, df_2])
    
    # get center
    center_dic = get_center()
    center_list = []
    for i in center_dic:
        center_list.append(center_dic[i])

    # add column ['sum'] with these 5 aspects
    df_train['sum'] = df_train['creation_date'] + df_train['update_date'] + df_train['expire_date'] + df_train['dic_score'] + df_train['mx_score'] 
    
    # sum of df_train['cluster_id'] == 0,1,2
    cluster_id_0 = df_train.loc[df_train['cluster_id'] == 0, 'sum'].sum()
    cluster_id_1 = df_train.loc[df_train['cluster_id'] == 1, 'sum'].sum()
    cluster_id_2 = df_train.loc[df_train['cluster_id'] == 2, 'sum'].sum()

    # avg of df_train['cluster_id'] == 0,1,2
    avg_cluster_id_0 = cluster_id_0/len(df_train[df_train['cluster_id'] == 0])
    avg_cluster_id_1 = cluster_id_1/len(df_train[df_train['cluster_id'] == 1])
    avg_cluster_id_2 = cluster_id_2/len(df_train[df_train['cluster_id'] == 2])

    # compare these 3 numbers
    compare = [avg_cluster_id_0, avg_cluster_id_1, avg_cluster_id_2]
    max_num = max(compare)
    min_num = min(compare)

    dic = {0: avg_cluster_id_0, 1: avg_cluster_id_1, 2: avg_cluster_id_2}
    tag = {}
    for i in dic:
        if max_num == dic[i]:
            tag['max'] = i
        elif min_num == dic[i]:
            tag['min'] = i
        else:
            tag['mid'] = i

    def assign_label(i):
        if i == tag['max']:
            return 'very suspicious'
        if i == tag['min']:
            return 'unsuspicious'
        if i == tag['mid']:
            return 'likely suspicious'

    df_train['prediction'] = df_train['cluster_id'].apply(assign_label)

    # classifying testing data
    # k-classifying
    classify_result = classifying_data(test_data, center_list)

    df_test = pd.DataFrame(columns=['creation_date', 'update_date', 'expire_date', 'dic_score', 'mx_score'])

    for i, item in enumerate(test_data):
        df_test.loc[i] = item
        
    # add cluster label to test dataframe
    df_test['cluster_id'] = list(classify_result.values())

    # add prediction tag to test dataframe
    df_test['prediction'] = df_test['cluster_id'].apply(assign_label)

    # grab ONE test data point to pass to java team
    json_data = {'prediction': df_test['prediction'][0], 'dic_score': df_test['dic_score'][0], 'mx_score': df_test['mx_score'][0]}
    
    # 以 json 格式回傳預測
    return json.dumps(json_data, ensure_ascii=False) 


if __name__ == '__main__':
#--test--
    domain = 'apple.com'

    # input domain to get dic_score
    dic_score = int(fit.detect_word_not_in_dictionary(domain))

    # input domain to get mx_score
    mx_score = int(fit.mx_score(domain))

    # test_data: combine all scores
    test_data = [[0, 8, 10, dic_score, mx_score]]

    json_data = test_result(test_data)
    print(json_data)
    # print(test_result(json_data_from_java["domain"], test_data))
    #                   json_data_from_java["score2"],
    #                   json_data_from_java["score3"]))
#--test--

    # data_from_java = []
    # for i in range(1, len(sys.argv)):

    #     data_from_java.append((sys.argv[i]))

    # word_not_in_dictionary_score = detect_word_not_in_dictionary(data_from_java[0])
    # print(train_and_predict(data_from_java[0], 
    #                   word_not_in_dictionary_score, 
    #                   data_from_java[1],
    #                   data_from_java[2],
    #                   data_from_java[3]))