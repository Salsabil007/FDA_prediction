import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import tensorflow as tf
from keras.regularizers import L1L2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
import random
random.seed(10)
from sklearn.linear_model import LogisticRegression

'''In this file, we are creating feature of prediction in y-1, y-2, y-3 year for each cluster in the training set'''
def training(x_train):
    y_train = x_train['class']
    x_train = x_train.drop(['cluster','n','year','pct_dusted_ccn','pct_dusted_rmcl','is_clinical','cited_by_clin','animal','cluster','year','class','pct_is_newish','pct_of_biggest_anc_new',
    'pct_of_biggest_anc_newish','rcr_mid','pct_of_biggest_anc','pct_in_secbiggest_anc','pct_of_secbiggest_anc','pct_of_secbiggest_anc_new','pct_of_secbiggest_anc_newish',
    'biggest_anc','pct_in_biggest_anc','secbiggest_anc'], axis = 1)


    x_train[['wrcr','rcr_hi','rcr_low','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc']] = minmax_scale(x_train[['wrcr','rcr_hi','rcr_low','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc']])
    print("data type training")
    print(x_train.dtypes)
    y_train = np.array(y_train)
    x_train = x_train.values
    model = LogisticRegression(random_state=0).fit(x_train,y_train)
    pred = model.predict(x_train)
    print("f1 score ",f1_score(y_train, pred , average="macro"))
    return model

def find_pct(data):
    pctisnew_1,pctisnew_2,pctisnew_3,pctisnew_4,pctisnew_5,pctisnew_6,pctisnew_7 = [],[],[],[],[],[],[]
    for ind in data.index:
        y0 = int(data['year'][ind])
        an0 = data['biggest_anc'][ind]
        fl = str(y0-1) + "_cluster_stability.csv"
        dfl = pd.read_csv(fl)
        row = dfl[dfl.cluster == an0]
        if len(row) >= 1:
            row = row.values
            y0 = int(row[0][1])
            an0 = row[0][19]
            pctisnew_1.append(row[0][3])
            
        else:
            pctisnew_1.append(-1)
        

        fl = str(y0-1) + "_cluster_stability.csv"
        dfl = pd.read_csv(fl)
        row = dfl[dfl.cluster == an0]
        if len(row) >= 1:
            row = row.values
            y0 = int(row[0][1])
            an0 = row[0][19]
            pctisnew_2.append(row[0][3])
        
        else:
            pctisnew_2.append(-1)
        

        fl = str(y0-1) + "_cluster_stability.csv"
        dfl = pd.read_csv(fl)
        row = dfl[dfl.cluster == an0]
        if len(row) >= 1:
            row = row.values
            y0 = int(row[0][1])
            an0 = row[0][19]
            pctisnew_3.append(row[0][3])
        
        else:
            pctisnew_3.append(-1)
        

        fl = str(y0-1) + "_cluster_stability.csv"
        dfl = pd.read_csv(fl)
        row = dfl[dfl.cluster == an0]
        if len(row) >= 1:
            row = row.values
            y0 = int(row[0][1])
            an0 = row[0][19]
            pctisnew_4.append(row[0][3])
        
        else:
            pctisnew_4.append(-1)
        

        fl = str(y0-1) + "_cluster_stability.csv"
        dfl = pd.read_csv(fl)
        row = dfl[dfl.cluster == an0]
        if len(row) >= 1:
            row = row.values
            y0 = int(row[0][1])
            an0 = row[0][19]
            pctisnew_5.append(row[0][3])
        
        else:
            pctisnew_5.append(-1)
    

        fl = str(y0-1) + "_cluster_stability.csv"
        dfl = pd.read_csv(fl)
        row = dfl[dfl.cluster == an0]
        if len(row) >= 1:
            row = row.values
            y0 = int(row[0][1])
            an0 = row[0][19]
            pctisnew_6.append(row[0][3])
        
        else:
            pctisnew_6.append(-1)
        

        fl = str(y0-1) + "_cluster_stability.csv"
        dfl = pd.read_csv(fl)
        row = dfl[dfl.cluster == an0]
        if len(row) >= 1:
            row = row.values
            y0 = int(row[0][1])
            an0 = row[0][19]
            pctisnew_7.append(row[0][3])
        
        else:
            pctisnew_7.append(-1)
        

    data['pctisnew_1'] = pctisnew_1
    data['pctisnew_2'] = pctisnew_2
    data['pctisnew_3'] = pctisnew_3
    data['pctisnew_4'] = pctisnew_4
    data['pctisnew_5'] = pctisnew_5
    data['pctisnew_6'] = pctisnew_6
    data['pctisnew_7'] = pctisnew_7
    



    ff = 0
    for ind in data.index:
        if data['pctisnew_7'][ind] == -1:
            data['pctisnew_7'][ind] = 0.00 
            ff = 1
        if data['pctisnew_6'][ind] == -1:
            data['pctisnew_6'][ind] = data['pctisnew_7'][ind]
        if data['pctisnew_5'][ind] == -1:
            data['pctisnew_5'][ind] = (data['pctisnew_7'][ind] + data['pctisnew_6'][ind])/2.00 
        if data['pctisnew_4'][ind] == -1:
            data['pctisnew_4'][ind] = (data['pctisnew_7'][ind] + data['pctisnew_6'][ind] + data['pctisnew_5'][ind])/3.00 
        if data['pctisnew_3'][ind] == -1:
            data['pctisnew_3'][ind] = (data['pctisnew_7'][ind] + data['pctisnew_6'][ind] + data['pctisnew_5'][ind] + data['pctisnew_4'][ind])/4.00 
        if data['pctisnew_2'][ind] == -1:
            data['pctisnew_2'][ind] = (data['pctisnew_7'][ind] + data['pctisnew_6'][ind] + data['pctisnew_5'][ind] + data['pctisnew_4'][ind] + data['pctisnew_3'][ind])/5.00
        if data['pctisnew_1'][ind] == -1:
            data['pctisnew_1'][ind] = (data['pctisnew_7'][ind] + data['pctisnew_6'][ind] + data['pctisnew_5'][ind] + data['pctisnew_4'][ind] + data['pctisnew_3'][ind] + data['pctisnew_2'][ind])/6.00 
        if ff == 1:
            data['pctisnew_7'][ind] = (data['pctisnew_1'][ind] + data['pctisnew_6'][ind] + data['pctisnew_5'][ind] + data['pctisnew_4'][ind] + data['pctisnew_3'][ind] + data['pctisnew_2'][ind])/6.00
    
    return data

def match(proba, data, yr):
    prev = []
    pprev = []
    ppprev = []
    #print(proba)
    for ind in data.index:
        x = proba[proba.c2 == data["cluster"][ind]]
        #print(x)
        x1 = x[x.year == yr-1]
        if len(x1) == 0:
            prev.append(0)
        else:
            x1 = x1["pred_log"]
            x1 = x1.to_numpy()
            prev.append(x1[0])
        x2 = x[x.year == yr-2]
        if len(x2) == 0:
            pprev.append(0)
        else:
            x2 = x2["pred_log"]
            x2 = x2.to_numpy()
            pprev.append(x2[0])
        x3 = x[x.year == yr-3]
        if len(x3) == 0:
            ppprev.append(0)
        else:
            x3 = x3["pred_log"]
            x3 = x3.to_numpy()
            ppprev.append(x3[0])
    data["prev"] = prev
    data["pprev"] = pprev
    data["ppprev"] = ppprev
    return data
    
def testing(x_test, model):
    #print(x_test.dtypes)
    column_means = x_test.mean()
    x_test = x_test.fillna(column_means)
    temp = x_test
    x_test = x_test.drop(['c2','prediction','year','cluster','biggest_anc','pct_in_biggest_anc','secbiggest_anc','is_clinical','cited_by_clin','pct_is_newish'], axis=1)
   
    x_test[['wrcr','rcr_hi','rcr_low','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc']] = minmax_scale(x_test[['wrcr','rcr_hi','rcr_low','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc']])
    x_test = x_test.drop(['pct_of_biggest_anc_new'], axis=1)
    print("datatype test")
    print(x_test.dtypes)
    x_test = x_test.values
    predy = model.predict_proba(x_test)
    pred_test = []
    for i in range(predy.shape[0]):
        if predy[i][0] >= predy[i][1]:
            #pred_test.append(0.00)
            pred_test.append(predy[i][1])
        else:
            pred_test.append(predy[i][1])
    temp['pred_log'] = pred_test
    temp = temp.sort_values(by=['pred_log'],ascending=False,ignore_index=True)
    return temp


x_train = pd.read_csv("data_84.csv")
x_train = x_train
#print(len(x_train))

yrs = x_train["year"].unique()
#print(yrs)
train_final = pd.DataFrame()


for i in yrs:
    model = training(x_train[x_train.year != i])
    tst = x_train[x_train.year == i]
    pp = tst
    tst = tst["cluster"].unique()
    #print(tst)

    df = pd.DataFrame()
    
    for j in tst:
        big_anc_j = pp[pp.cluster == j]
        big_anc_j = big_anc_j.to_numpy()
        print("for year ",i," node ",j)
        bj = big_anc_j[0][7]
        print("anc 1 ",bj)
        if i == 1979:
            d1 = pd.read_csv("features_prediction_"+str(i-1)+".csv")
            d1 = d1[d1.cluster == bj]
            d1['c2'] = j
            df = df.append(d1, ignore_index= True)
            
        elif i == 1980:
            d1 = pd.read_csv("features_prediction_"+str(i-1)+".csv")
            d1 = d1[d1.cluster == bj]

            big_anc_j = d1
            big_anc_j = big_anc_j.to_numpy()
            bj = big_anc_j[0][5]
            print("anc 2 ",bj)
            d2 = pd.read_csv("features_prediction_"+str(i-2)+".csv")
            d2 = d2[d2.cluster == bj] ######################## d2 = d2[d2.cluster == j] 
            d1 = d1.append(d2, ignore_index= True)
            d1['c2'] = j
            df = df.append(d1, ignore_index= True)
        else:

            d1 = pd.read_csv("features_prediction_"+str(i-1)+".csv")
            d1 = d1[d1.cluster == bj]

            big_anc_j = d1
            big_anc_j = big_anc_j.to_numpy()
            bj = big_anc_j[0][5]
            d2 = pd.read_csv("features_prediction_"+str(i-2)+".csv")
            d2 = d2[d2.cluster == bj]
            print("anc 2 ",bj)

            big_anc_j = d2
            big_anc_j = big_anc_j.to_numpy()
            bj = big_anc_j[0][5]
            d3 = pd.read_csv("features_prediction_"+str(i-3)+".csv")
            d3 = d3[d3.cluster == bj]
            print("anc 3 ",bj)

            d1 = d1.append(d2, ignore_index= True)
            d1 = d1.append(d3, ignore_index= True)
            d1['c2'] = j
            df = df.append(d1, ignore_index= True)

        df = df.drop(['n','pct_of_biggest_anc_newish','rcr_mid',
        'pct_of_biggest_anc','pct_in_secbiggest_anc','pct_of_secbiggest_anc','pct_of_secbiggest_anc_new','pct_of_secbiggest_anc_newish','animal'], axis = 1)
        
    '''if i > 1986:
        df = find_pct(df)
    else:
        df['pctisnew_1'] = 0.00
        df['pctisnew_2'] = 0.00
        df['pctisnew_3'] = 0.00
        df['pctisnew_4'] = 0.00
        df['pctisnew_5'] = 0.00
        df['pctisnew_6'] = 0.00
        df['pctisnew_7'] = 0.00'''

    #df = testing(df)
    proba = testing(df,model)
    #proba.to_csv("int.csv", index = False)
    train_temp = match(proba, x_train[x_train.year == i], i)
    train_final = train_final.append(train_temp, ignore_index= True)
    
train_final.to_csv("training_prev_year2.csv", index = False)
    
    
