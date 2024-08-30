from unittest import result
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys 
import math

#from keras.layers import Dense
#from keras.utils import np_utils
import tensorflow as tf
import tensorflow.keras
#from keras.models import Sequential
#from keras.regularizers import L1L2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#from keras.callbacks import EarlyStopping
import random
random.seed(10)
from sklearn.linear_model import LogisticRegression

def cutoff(data):
    #print("length before cutoff ", len(data))
    data = data[data.n > 20]
    data = data[data['is_research_article'] > 0.75]
    #data = data[data['nih'] > 0.05]
    sum_col = data['human']+data['animal']+data['molecular_cellular']
    data['sum'] = sum_col
    data = data[data['sum'] > 0.75]
    data = data.drop(['sum'], axis = 1)
    #print("length after cutoff ", len(data))
    return data

def extreme_growth(year,h):
    data = pd.read_csv("with250_previous_years_"+str(year)+".csv")
    data = cutoff(data)
    print(data.dtypes)
    df = pd.DataFrame()
    df['cluster'] = data['cluster']
    df['n'] = data['n']
    df['top_250'] = np.log10(data['counts']+1)

    tt = pd.read_csv("paper_vit2_"+str(year)+".csv")
    tt = tt.rename(columns = {'cl':'cluster'})
    tt = tt.drop(['pmid','year','age'], axis = 1)

    result1 = pd.merge(df,tt,on='cluster', how = 'left')
    result1['paper_vitality'] = result1['val']/result1['n']
    result1 = result1.drop(['val'], axis = 1)


    tt = pd.read_csv("growth_year_"+str(year)+".csv")
    tt = tt.drop(['peak_year'], axis = 1)
    result2 = pd.merge(result1,tt,on='cluster', how = 'left')

    tt = pd.read_csv("cit_vit_"+str(year)+".csv")
    tt = tt.rename(columns = {'cl':'cluster'})
    result3 = pd.merge(result2,tt,on='cluster', how = 'left')
    result3['citation_vitality'] = np.power((result3['referenced_year']/result3['n']), (1/4))
    result3 = result3.drop(['referenced_year'], axis = 1)

    result3['paper_vitality'] = result3['paper_vitality'] - result3['paper_vitality'].mean()
    result3['paper_vitality'] = result3['paper_vitality']/result3['paper_vitality'].std()
    result3['top_250'] = result3['top_250'] - result3['top_250'].mean()
    result3['top_250'] = result3['top_250']/result3['top_250'].std()
    result3['growth_stage'] = result3['growth_stage'] - result3['growth_stage'].mean()
    result3['growth_stage'] = result3['growth_stage']/result3['growth_stage'].std()
    result3['citation_vitality'] = result3['citation_vitality'] - result3['citation_vitality'].mean()
    result3['citation_vitality'] = result3['citation_vitality']/result3['citation_vitality'].std()

    result3['pred_log'] = 0.473 * result3['paper_vitality']  + 0.292 * result3['growth_stage'] + 0.1 * result3['citation_vitality'] + 0.113 * result3['top_250']
    result3 = result3.sort_values(by=['pred_log'], ascending=False)
    result3 = result3.head(h)

    data2 = pd.read_csv("with250_training_prev_year3.csv")
    data2 = data2[data2.n > 20]
    data2 = data2[data2['is_research_article'] > 0.75]

    sum_col = data2['human']+data2['animal']+data2['molecular_cellular']
    data2['sum'] = sum_col
    data2 = data2[data2['sum'] > 0.75]
    data2 = data2.drop(['sum'], axis = 1)

    d2 = pd.DataFrame()
    d2['cluster'] = data2['cluster']
    d2['class'] = data2['class']
    d2['year'] = data2['year']
    d2 = d2[d2.year == year]
    
    result = pd.merge(result3,d2,on=["cluster"], how = "inner")
    print("actual positive for year " + str(year) +" is ", len(d2[d2["class"] == 1]))
    print("no of true positive in extreme growth ",len(result[result['class'] == 1]))
    print("no of false positive in extreme growth that are in training set ",len(result[result['class'] == 0]))
    return len(result[result['class'] == 1]),len(d2[d2["class"] == 1]),len(result[result['class'] == 0])

y = 1984
sum = 0
total = 0
fn = 0
i = 0
#pos = [95, 85, 98, 105, 127, 132, 150, 147, 146, 143, 143, 189, 166, 154, 135, 144, 142, 95, 137, 155, 129, 128, 149, 128, 88, 106, 104, 141, 99, 113, 68, 91, 111, 78]
while y<=2017:
    a,b, c = extreme_growth(y,155)
    sum += a
    total += b
    fn += c
    y += 1
    i += 1

print(sum)
print(total)
##get the average no of positives for our model and use that as positive cutoff for the extreme growth model