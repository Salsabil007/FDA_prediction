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
    data = data[data.n > 50]
    #data = data[data['is_research_article'] > 0.75]
    #data = data[data['nih'] > 0.05]
    sum_col = data['human']+data['animal']+data['molecular_cellular']
    data['sum'] = sum_col
    #data = data[data['sum'] > 0.75]
    data = data.drop(['sum'], axis = 1)
    #print("length after cutoff ", len(data))
    return data


x_test = pd.read_csv("with250_test.csv")
x_test = cutoff(x_test)


df = pd.DataFrame()
df['cluster'] = x_test['cluster']
df['n'] = x_test['n']
df['intersect'] = x_test['intersect']
df['pct_covid'] = x_test['pct_covid']
df['top_250'] = np.log10(x_test['counts']+1)

tt = pd.read_csv("paper_vit2_2020.csv")
tt = tt.rename(columns = {'cl':'cluster'})
tt = tt.drop(['pmid','year','age'], axis = 1)

result1 = pd.merge(df,tt,on='cluster', how = 'left')
result1['paper_vitality'] = result1['val']/result1['n']
result1 = result1.drop(['val'], axis = 1)


tt = pd.read_csv("growth_year_2020.csv")
tt = tt.drop(['peak_year'], axis = 1)
result2 = pd.merge(result1,tt,on='cluster', how = 'left')

tt = pd.read_csv("cit_vit_2020.csv")
tt = tt.rename(columns = {'cl':'cluster'})
result3 = pd.merge(result2,tt,on='cluster', how = 'left')
result3['citation_vitality'] = np.power((result3['citing_year']/result3['n']), (1/4))
result3 = result3.drop(['citing_year'], axis = 1)

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
#print(result3.head(194))

'''print(len(result2))
print(len(tt))
print(len(result3))
print(result3.head(5))
'''




true_covid = result3[result3.pct_covid > 0.34]
true_covid = true_covid[true_covid.intersect > 50]
print("len of true covid points ", len(true_covid))

result3 = result3.head(92)
pred_covid = result3[result3.pct_covid > 0.34]
pred_covid = pred_covid[pred_covid.intersect > 50]
print("len of predicted covid points ", len(pred_covid))
