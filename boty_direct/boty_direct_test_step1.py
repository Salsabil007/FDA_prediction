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

'''In this part, we are trying to create features for training. I will only keep the following features:
(pct_is_new, pct_of_biggest_anc_new, nih, rcr_low, rcr_mid, wrcr, human, molecular_cellular) and will go back 7 years for each of them. 
we will also keep n_biggest_anc and n_secbiggest_anc but won't go backward.
'''

data = pd.read_csv("2020_overlap.csv")
'''data = data.drop(['n','pct_of_biggest_anc_newish','rcr_mid','secbiggest_anc',
'pct_of_biggest_anc','pct_in_secbiggest_anc','pct_of_secbiggest_anc','pct_of_secbiggest_anc_new','pct_of_secbiggest_anc_newish','n_clusts_90_anc','pct_top_5_anc','animal','is_clinical',
'cited_by_clin'], axis = 1)'''
#print(data.isnull().sum())
#data = data.dropna()
data = data.fillna(data.mean())
#print(data.isnull().sum())
data['biggest_anc'] = data['biggest_anc'].astype(int)
data['n_biggest_anc'] = data['n_biggest_anc'].astype(int)
data['n_secbiggest_anc'] = data['n_secbiggest_anc'].astype(int)
print(data.dtypes)

pctisnew_1,pctisnew_2,pctisnew_3,pctisnew_4,pctisnew_5,pctisnew_6,pctisnew_7 = [],[],[],[],[],[],[]
pctobnew_1,pctobnew_2,pctobnew_3,pctobnew_4,pctobnew_5,pctobnew_6,pctobnew_7 = [],[],[],[],[],[],[]
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
        pctobnew_1.append(row[0][23])
    else:
        pctisnew_1.append(-1)
        pctobnew_1.append(-1)

    fl = str(y0-1) + "_cluster_stability.csv"
    dfl = pd.read_csv(fl)
    row = dfl[dfl.cluster == an0]
    if len(row) >= 1:
        row = row.values
        y0 = int(row[0][1])
        an0 = row[0][19]
        pctisnew_2.append(row[0][3])
        pctobnew_2.append(row[0][23])
    else:
        pctisnew_2.append(-1)
        pctobnew_2.append(-1)

    fl = str(y0-1) + "_cluster_stability.csv"
    dfl = pd.read_csv(fl)
    row = dfl[dfl.cluster == an0]
    if len(row) >= 1:
        row = row.values
        y0 = int(row[0][1])
        an0 = row[0][19]
        pctisnew_3.append(row[0][3])
        pctobnew_3.append(row[0][23])
    else:
        pctisnew_3.append(-1)
        pctobnew_3.append(-1)

    fl = str(y0-1) + "_cluster_stability.csv"
    dfl = pd.read_csv(fl)
    row = dfl[dfl.cluster == an0]
    if len(row) >= 1:
        row = row.values
        y0 = int(row[0][1])
        an0 = row[0][19]
        pctisnew_4.append(row[0][3])
        pctobnew_4.append(row[0][23])
    else:
        pctisnew_4.append(-1)
        pctobnew_4.append(-1)

    fl = str(y0-1) + "_cluster_stability.csv"
    dfl = pd.read_csv(fl)
    row = dfl[dfl.cluster == an0]
    if len(row) >= 1:
        row = row.values
        y0 = int(row[0][1])
        an0 = row[0][19]
        pctisnew_5.append(row[0][3])
        pctobnew_5.append(row[0][23])
    else:
        pctisnew_5.append(-1)
        pctobnew_5.append(-1)

    fl = str(y0-1) + "_cluster_stability.csv"
    dfl = pd.read_csv(fl)
    row = dfl[dfl.cluster == an0]
    if len(row) >= 1:
        row = row.values
        y0 = int(row[0][1])
        an0 = row[0][19]
        pctisnew_6.append(row[0][3])
        pctobnew_6.append(row[0][23])
    else:
        pctisnew_6.append(-1)
        pctobnew_6.append(-1)

    fl = str(y0-1) + "_cluster_stability.csv"
    dfl = pd.read_csv(fl)
    row = dfl[dfl.cluster == an0]
    if len(row) >= 1:
        row = row.values
        y0 = int(row[0][1])
        an0 = row[0][19]
        pctisnew_7.append(row[0][3])
        pctobnew_7.append(row[0][23])
    else:
        pctisnew_7.append(-1)
        pctobnew_7.append(-1)

data['pctisnew_1'] = pctisnew_1
data['pctisnew_2'] = pctisnew_2
data['pctisnew_3'] = pctisnew_3
data['pctisnew_4'] = pctisnew_4
data['pctisnew_5'] = pctisnew_5
data['pctisnew_6'] = pctisnew_6
data['pctisnew_7'] = pctisnew_7

data['pctobnew_1'] = pctobnew_1
data['pctobnew_2'] = pctobnew_2
data['pctobnew_3'] = pctobnew_3
data['pctobnew_4'] = pctobnew_4
data['pctobnew_5'] = pctobnew_5
data['pctobnew_6'] = pctobnew_6
data['pctobnew_7'] = pctobnew_7

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

ff = 0
for ind in data.index:
    if data['pctobnew_7'][ind] == -1:
        data['pctobnew_7'][ind] = 0.00 
        ff = 1
    if data['pctobnew_6'][ind] == -1:
        data['pctobnew_6'][ind] = data['pctobnew_7'][ind]
    if data['pctobnew_5'][ind] == -1:
        data['pctobnew_5'][ind] = (data['pctobnew_7'][ind] + data['pctobnew_6'][ind])/2.00 
    if data['pctobnew_4'][ind] == -1:
        data['pctobnew_4'][ind] = (data['pctobnew_7'][ind] + data['pctobnew_6'][ind] + data['pctobnew_5'][ind])/3.00 
    if data['pctobnew_3'][ind] == -1:
        data['pctobnew_3'][ind] = (data['pctobnew_7'][ind] + data['pctobnew_6'][ind] + data['pctobnew_5'][ind] + data['pctobnew_4'][ind])/4.00 
    if data['pctobnew_2'][ind] == -1:
        data['pctobnew_2'][ind] = (data['pctobnew_7'][ind] + data['pctobnew_6'][ind] + data['pctobnew_5'][ind] + data['pctobnew_4'][ind] + data['pctobnew_3'][ind])/5.00
    if data['pctobnew_1'][ind] == -1:
        data['pctobnew_1'][ind] = (data['pctobnew_7'][ind] + data['pctobnew_6'][ind] + data['pctobnew_5'][ind] + data['pctobnew_4'][ind] + data['pctobnew_3'][ind] + data['pctobnew_2'][ind])/6.00 
    if ff == 1:
        data['pctobnew_7'][ind] = (data['pctobnew_1'][ind] + data['pctobnew_6'][ind] + data['pctobnew_5'][ind] + data['pctobnew_4'][ind] + data['pctobnew_3'][ind] + data['pctobnew_2'][ind])/6.00

data.to_csv("test_allfeature_boty.csv", index = False)