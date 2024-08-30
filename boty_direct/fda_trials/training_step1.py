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

##this file is taking the raw trainign dataset and shuffle and check correlation. it is the first part.


data = pd.read_csv("gold_control_datapoints_prized_samp_fda.csv")
data = data.sort_values(by=['cluster'])
data = shuffle(data)
data = shuffle(data)
data = shuffle(data)
data = shuffle(data)
data = shuffle(data)
data = shuffle(data)
data = shuffle(data)
data = shuffle(data)
data = shuffle(data)
data = shuffle(data)
print(len(data))
data.dropna(axis=0, inplace= True)
print(len(data))
data['class'] = data['class'].astype(int)
#print(data.dtypes)
#data = data[data.year >= 1984]
data = data.sort_values(by=['year'])
data.to_csv("data_84.csv", index = False)
print(len(data))

'''
data = pd.read_csv("data_84.csv")
sys.stdout = open("correlation_incites.txt", "w")
corr = data.corr()
print(corr)
corr_1 = corr.iloc[:,0:9]
print("printing correlation")
print(corr_1)
corr_2 = corr.iloc[:,9:16]
print("printing correlation")
print(corr_2)
corr_3 = corr.iloc[:,16:24]
print("printing correlation")
print(corr_3)
corr_4 = corr.iloc[:,24:]
print("printing correlation")
print(corr_4)
'''