import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys 

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
    data = data[data.n > 100]
    #data = data[data['is_research_article'] > 0.75]
    #data = data[data['nih'] > 0.05]
    sum_col = data['human']+data['animal']+data['molecular_cellular']
    data['sum'] = sum_col
    #data = data[data['sum'] > 0.75]
    data = data.drop(['sum'], axis = 1)
    #print("length after cutoff ", len(data))
    return data
def crossvalidation(x_train, y_train):
    cvscores = []
    f1score=[]
    Kfold = KFold(n_splits=5, random_state=1, shuffle=True)
    for train, test in Kfold.split(x_train, y_train):
        model = LogisticRegression(random_state=0, C = 0.40).fit(x_train,y_train)
        '''scores = model.evaluate(x_train[test], y_train[test], verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)'''
        pre = model.predict(x_train[test])
        prey = np.where(pre > 0.5, 1,0)
        f1score.append(f1_score(y_train[test], prey , average="macro"))
    #print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("mean f1 score: ", np.mean(f1score))

def fet(data):
    y = pd.DataFrame()
    y['pct_is_new'] = data['pct_is_new']
    #y['pct_is_newish'] = data['pct_is_newish']
    #y['pct_in_biggest_anc'] = data['pct_in_biggest_anc']
    #y['pct_of_biggest_anc'] = data['pct_of_biggest_anc']
    y['pct_of_biggest_anc_new'] = data['pct_of_biggest_anc_new']
    #y['pct_of_secbiggest_anc_new'] = data['pct_of_secbiggest_anc_new']
    y['n_clusts_90_anc'] = data['n_clusts_90_anc']
    y['pct_top_5_anc'] = data['pct_top_5_anc']
    y['rcr_low'] = data['rcr_low']
    y['rcr_mid'] = data['rcr_mid']
    y['rcr_hi'] = data['rcr_hi']
    y['human'] = data['human']
    y['is_research_article'] = data['is_research_article']
    y['rage'] = data['rage']
    y['cited_by_clin'] = data['cited_by_clin']

    y[['n_clusts_90_anc','pct_of_biggest_anc_new','human']] = minmax_scale(y[['n_clusts_90_anc','pct_of_biggest_anc_new','human']])
    return y

x_train = pd.read_csv("with250_training_prev_year3.csv")
x_test = pd.read_csv("with250_test_prev_year2.csv")
x_train = x_train[x_train.n > 100]
#x_train = x_train[x_train['is_research_article'] > 0.75]
lenn = len(x_train)
sum_col = x_train['human']+x_train['animal']+x_train['molecular_cellular']
x_train['sum'] = sum_col
#x_train = x_train[x_train['sum'] > 0.75]
x_train = x_train.drop(['sum'], axis = 1)
print("hdhdhd after ",len(x_train))
###removhe here
x_test = cutoff(x_test)

x_train = x_train.fillna(x_train.mean())
x_test = x_test.fillna(x_test.mean())

y_train = x_train['class']
x_train = x_train.drop(['class'], axis=1)

temp = x_test
x_test = x_test.drop(['prediction','intersect','pct_covid'], axis=1)

'''
##Code 1
x_train[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new']] = minmax_scale(x_train[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new']])
x_train = x_train.drop(['is_clinical','cited_by_clin','diffpctob','diffpct','cluster','year',"pctisnew_7",'pprev','pct_in_biggest_anc','pct_of_biggest_anc','rcr_low','pct_is_newish'], axis = 1)#$%




x_test[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new']] = minmax_scale(x_test[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new']])
x_test = x_test.drop(['n','pct_of_biggest_anc_newish','rcr_low','pct_of_biggest_anc','pct_in_secbiggest_anc','pct_of_secbiggest_anc','pct_of_secbiggest_anc_newish'], axis = 1)
x_test = x_test.drop(['year','cluster','biggest_anc','pct_in_biggest_anc','secbiggest_anc','is_clinical','cited_by_clin','pprev','pctobnew_1','pctobnew_2','pctobnew_3','pctobnew_4','pctobnew_5','pctobnew_6','pctobnew_7',
'pctisnew_1','pctisnew_2','pctisnew_3','pctisnew_4','pctisnew_5','pctisnew_6','pctisnew_7','diffpct','diffpctob','pct_is_newish'], axis=1) #$%
'''

'''
##Code 2
x_train = fet(x_train)
x_test = fet(x_test)'''

##Code 3
x_train[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new']] = minmax_scale(x_train[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new']])
x_train = x_train.drop(['n','cited_by_clin','diffpctob','diffpct','cluster','year',"pctisnew_7",'pprev','pct_in_biggest_anc','pct_of_biggest_anc','rcr_low'], axis = 1)#$%
#x_train = x_train.drop(['pct_of_biggest_anc_new','pct_of_secbiggest_anc_new','animal','pct_is_newish'], axis = 1)
x_train = x_train.drop(['counts','pct_is_newish','pct_of_biggest_anc_new'], axis = 1)

x_test[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new']] = minmax_scale(x_test[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new']])
x_test = x_test.drop(['n','pct_of_biggest_anc_newish','rcr_low','pct_of_biggest_anc','pct_in_secbiggest_anc','pct_of_secbiggest_anc','pct_of_secbiggest_anc_newish'], axis = 1)
x_test = x_test.drop(['year','cluster','biggest_anc','pct_in_biggest_anc','secbiggest_anc','cited_by_clin','pprev'], axis=1) #$%
#x_test = x_test.drop(['pct_of_biggest_anc_new','pct_of_secbiggest_anc_new','animal','pct_is_newish'], axis = 1)
x_test = x_test.drop(['counts','pct_is_newish','pct_of_biggest_anc_new','pctisnew_1','pctisnew_2','pctisnew_3','pctisnew_4','pctisnew_5','pctisnew_6','pctisnew_7','pctobnew_1','pctobnew_2','pctobnew_3','pctobnew_4','pctobnew_5','pctobnew_6','pctobnew_7','diffpct','diffpctob'], axis = 1)


y_train = np.array(y_train)
###print(x_train.dtypes)
x_train = x_train.values

model = LogisticRegression(random_state=0, C = 0.40).fit(x_train,y_train)

pred = model.predict(x_train)
###print(pred[:50])


###print("f1 score ",f1_score(y_train, pred , average="macro"))
crossvalidation(x_train, y_train)



###print("length of test set ", len(x_test))


###print(x_test.dtypes)
#print("%^&%^&%^&")
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

tt = temp[temp.pred_log > 0.5]
#print("total ", len(temp)," length of positive by LR ",len(tt)," percentage ",len(tt)/len(temp))
good = temp[temp.intersect > 50]
good = good[good.pct_covid > 0.34]
good_correct = good[good.pred_log > 0.5]
'''print("length of good data ",len(good))
print("length of correct prediction by LR ",len(good_correct))
print("percentage of correct prediction of good data by LR ", len(good_correct)/len(good))'''

past = good[good.prediction > 0.5]
###print("percentage of correct prediction by RF ", len(past)/len(good))
#print(len(temp[temp.prediction > 0.5])/len(temp))
###print("total positive by RF ", len(temp[temp.prediction > 0.5])," correct positive by RF ", len(past))

'''temp = temp.sort_values(by=['prediction'])
plt.scatter(temp['prediction'], temp['pct_covid'], s = 10)
plt.ylabel('pct_covid')
plt.xlabel('prediction probability by random forest')
#plt.plot(temp['prediction'], temp['pct_covid'])
plt.show()'''


temp = temp.sort_values(by=['pred_log'])
plt.scatter(temp['pred_log'], temp['pct_covid'], s = 10)
plt.ylabel('pct_covid')
plt.xlabel('prediction by logistic regression')
#plt.plot(temp['prediction'], temp['pct_covid'])
#plt.show()



true_covid = temp[temp.pct_covid > 0.34]
true_covid = true_covid[true_covid.intersect > 50]


th = 0.85
while th <= 0.89:
    true_byLR = true_covid[true_covid.pred_log > th]
    true_byRF = true_covid[true_covid.prediction > th]
    print("for threshold ",th," true covid data points ", len(true_covid)," recall ",len(true_byLR), " fp ",len(temp[temp.pred_log > th]) - len(true_byLR))
    th += 0.01
#print("number of true covid points detected with high probability by lR ", len(true_byLR)," % of true covid point detected correctly ",len(true_byLR)/len(true_covid))
#print("number of true covid points detected with high probability by RF ", len(true_byRF)," % of true covid point detected correctly ",len(true_byRF)/len(true_covid))

#print("number of high probability given by LR ", len(temp[temp.pred_log > 0.85]) - len(true_byLR), " % of positive prediction which is correct ",len(true_byLR)/len(temp[temp.pred_log > 0.85]))
#print("number of high probability given by RF ", len(temp[temp.prediction > 0.85]), " % of positive prediction which is correct ",len(true_byRF)/len(temp[temp.prediction > 0.85]))

'''set1 = temp[temp.pred_log > 0.5]
plt.scatter(set1['pred_log'], set1['pct_covid'], s = 10)
plt.ylabel('pct_covid')
plt.xlabel('prediction probability (> 0.5)')
plt.show()'''

set2 = temp[temp.pct_covid > 0.34]
set2 = set2.sort_values(by=['pct_covid','pred_log'])
'''sns.distplot(set2['pct_covid'], hist=False, rug=True,color = 'red')
sns.distplot(set2['pred_log'], hist=False, rug=True)
plt.show()'''

set2 = set2[set2.pred_log > 0.50]
set2 = set2.sort_values(by=['pred_log'])


#sns.distplot(target_2[['sepal length (cm)']], hist=False, rug=True)

set2 = set2[set2.pred_log > 0.50]
'''z = np.polyfit(set2['pred_log'], set2['pct_covid'], 1)
p = np.poly1d(z)
plt.plot(set2['pred_log'],p(set2['pred_log']),"r--")
plt.plot(set2['pred_log'], set2['pct_covid'])
#plt.set_title('Chris Gayle', fontsize=14)
plt.xlabel('pred_log')
plt.ylabel('pct_covid')
plt.show()'''

set2 = set2[set2.pred_log > 0.70]
set2[['pred_log','pct_covid']] = minmax_scale(set2[['pred_log','pct_covid']])
set2 = set2.sort_values(by=['pred_log'])
sns.distplot(set2['pct_covid'], hist=False, rug=True, color = 'red')
sns.distplot(set2['pred_log'], hist=False, rug=True)
#plt.show()
#print("in boty")

'''
pct_is_new                   float64
n_biggest_anc                float64
pct_of_biggest_anc_new       float64
n_secbiggest_anc             float64
pct_of_secbiggest_anc_new    float64
n_clusts_90_anc              float64
pct_top_5_anc                float64
wrcr                         float64
rcr_mid                      float64
rcr_hi                       float64
human                        float64
animal                       float64
molecular_cellular           float64
is_research_article          float64
nih                          float64
rage                         float64
prev                         float64
ppprev                       float64
dtype: object
%^&%^&%^&

C: 0.60, th: 0.85, tp: 66, fp: 329
Code 1
'''

'''
For Code 2://fet
C = 0.50, th = 0.85, tp: 67, fp: 305
Code 2
'''

'''
Code 3://the one similar to without_pct2.py
C = 0.50, th = 0.85, tp: 68, fp: 211, CV_f1: 0.8418
C: 0.31, th = 0.85, tp: 67, fp: 148, CV_f1: 0.8317
cutoff: C= 0.38, th = 0.85, tp: 63/64, fp: 35, CV_f1: 0.8357
'''