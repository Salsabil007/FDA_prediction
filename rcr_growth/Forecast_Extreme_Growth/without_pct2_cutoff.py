import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, minmax_scale
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys 
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.utils import np_utils
import tensorflow as tf
#from keras.regularizers import L1L2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import shuffle
#from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np

np.random.seed(19680801)
def newfet_test(data,year):
    tt = pd.read_csv("paper_vit2_"+str(year)+".csv")
    tt = tt.rename(columns = {'cl':'cluster'})
    tt = tt.drop(['pmid','year','age'], axis = 1)

    result1 = pd.merge(data,tt,on='cluster', how = 'left')
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
    #result3['counts'] = result3['counts'] - result3['counts'].mean()
    #result3['counts'] = result3['counts']/result3['counts'].std()
    result3['growth_stage'] = result3['growth_stage'] - result3['growth_stage'].mean()
    result3['growth_stage'] = result3['growth_stage']/result3['growth_stage'].std()
    result3['citation_vitality'] = result3['citation_vitality'] - result3['citation_vitality'].mean()
    result3['citation_vitality'] = result3['citation_vitality']/result3['citation_vitality'].std()

    return result3

def newfeatureset():
    data = pd.read_csv("with250_training_prev_year3.csv")
    paper_vitality = []
    growth_stage = []
    citation_vitality = []
    for ind in data.index:
        y = data['year'][ind]
        cl = data['cluster'][ind]

        tt = pd.read_csv("paper_vit2_"+str(y)+".csv")
        tt = tt.rename(columns = {'cl':'cluster'})
        tt = tt.drop(['pmid','year','age'], axis = 1)
        tt = tt[tt['cluster'] == cl]
        tt = tt.to_numpy()
        paper_vitality.append(tt[0][1]/data['n'][ind])

        tt = pd.read_csv("growth_year_"+str(y)+".csv")
        tt = tt.drop(['peak_year'], axis = 1)
        tt = tt[tt['cluster'] == cl]
        tt = tt.to_numpy()
        growth_stage.append(tt[0][1])

        tt = pd.read_csv("cit_vit_"+str(y)+".csv")
        tt = tt.rename(columns = {'cl':'cluster'})
        tt = tt[tt['cluster'] == cl]
        tt = tt.to_numpy()
        citation_vitality.append(np.power((tt[0][1]/data['n'][ind]), (1/4)))
        
    data['paper_vitality'] = paper_vitality
    data['growth_stage'] = growth_stage
    data['citation_vitality'] = citation_vitality


    data['paper_vitality'] = data['paper_vitality'] - data['paper_vitality'].mean()
    data['paper_vitality'] = data['paper_vitality']/data['paper_vitality'].std()
    data['growth_stage'] = data['growth_stage'] - data['growth_stage'].mean()
    data['growth_stage'] = data['growth_stage']/data['growth_stage'].std()
    #data['counts'] = data['counts'] - data['counts'].mean()
    #data['counts'] = data['counts']/data['counts'].std()
    data['citation_vitality'] = data['citation_vitality'] - data['citation_vitality'].mean()
    data['citation_vitality'] = data['citation_vitality']/data['citation_vitality'].std()
    data.to_csv("extreme_training.csv", index = False)
    return data

def cutoff(data):
    print("length before cutoff ", len(data))
    data = data[data.n > 20]
    data = data[data['is_research_article'] > 0.75]
    #data = data[data['nih'] > 0.05]
    sum_col = data['human']+data['animal']+data['molecular_cellular']
    data['sum'] = sum_col
    data = data[data['sum'] > 0.75]
    data = data.drop(['sum'], axis = 1)
    print("length after cutoff ", len(data))
    return data
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
def calc_recall(y,data,th):

    #data = pd.read_csv(str(y)+"_allcluster.csv")
    print(len(data))
    true_high = data[data.isgold == "TRUE"]
    true_high = true_high[true_high.pred_log > th] ##gold points with high prediction probability

    false_high = data[data.isgold == "FALSE"] ## not gold points with high prediction probability
    false_high = false_high[false_high.pred_log > th]

    true_low = data[data.isgold == "TRUE"]
    true_low = true_low[true_low.pred_log <= th] ##gold points with low prediction probability

    
    data1 = true_low
    d1 = pd.DataFrame()
    d1['cluster'] = data1['cluster']
    d1['pred_log'] = data1['pred_log']

    ####data2 = pd.read_csv("with250_training_prev_year3.csv")
    data2 = pd.read_csv("extreme_training.csv")
    data2 = data2[data2.n > 20]
    data2 = data2[data2['is_research_article'] > 0.75]
    #data = data[data['nih'] > 0.05]
    sum_col = data2['human']+data2['animal']+data2['molecular_cellular']
    data2['sum'] = sum_col
    data2 = data2[data2['sum'] > 0.75]
    data2 = data2.drop(['sum'], axis = 1)
    d2 = pd.DataFrame()
    d2['cluster'] = data2['cluster']
    d2['class'] = data2['class']
    d2['year'] = data2['year']
    d2 = d2[d2.year == y]

    print("len of true low ", len(d1))

    result = pd.merge(d1,d2,on=["cluster"], how = "inner")
    print("len of result ", len(result))

    data3 = true_high
    d3 = pd.DataFrame()
    d3['cluster'] = data3['cluster']
    d3['pred_log'] = data3['pred_log']
    result2 = pd.merge(d3,d2,on=["cluster"], how = "inner")
    print("len of d3 ", len(d3)," len of result2 ", len(result2))
    r1 = result[result["class"] == 0]
    r2 = result2[result2["class"] == 1]

    print(str(y)+" total data points: ",len(d2))
    print(str(y)+" no of correctly classified data points: ",len(r1)+len(r2))
    print(str(y)+" false positive: ", len(data[data.pred_log > th])-len(r2))
    #print(str(y)+" false negative: ", len(d2[d2["class"] == 0])-len(r1))

    data4 = true_low
    d4 = pd.DataFrame()
    d4['cluster'] = data4['cluster']
    d4['pred_log'] = data4['pred_log']
    res = pd.merge(d4,d2,on=["cluster"], how = "inner")
    print("len of d4 ", len(d4)," len of res ", len(res))
    fn = res[res["class"] == 1]
    print(str(y)+" false negative: ", len(fn))
    
    print("length of data ", len(data))
    return len(r1)+len(r2), len(data[data.pred_log > th])-len(r2), len(fn), len(result2[result2["class"] == 0]),len(r2),len(data[data.pred_log > th]), len(d2[d2['class'] == 1])

def train(year,th):
    #year = 1984
    ####x_train = pd.read_csv("with250_training_prev_year3.csv")
    x_train = pd.read_csv("extreme_training.csv")
    print("hdhdhd before ",len(x_train))
    x_train = x_train[x_train.n > 20]
    x_train = x_train[x_train['is_research_article'] > 0.75]
    lenn = len(x_train)
    sum_col = x_train['human']+x_train['animal']+x_train['molecular_cellular']
    x_train['sum'] = sum_col
    x_train = x_train[x_train['sum'] > 0.75]
    x_train = x_train.drop(['sum'], axis = 1)
    x_train['counts'] = np.log10(x_train['counts']+1)
    
    ####x_train = newfeatureset()

    print("hdhdhd ",len(x_train))
    gold = x_train[x_train.year == year]
    x_train = x_train[x_train.year != year]
    #print(gold)
    print(len(x_train))
    #x_test = pd.read_csv("test_newfature.csv")
    #x_train = x_train.drop(['pctobnew_1','pctobnew_2','pctobnew_3','pctobnew_4','pctobnew_5','pctobnew_6'], axis=1)
    #drop some
    #x_test = x_test.drop(['pctobnew_1','pctobnew_2','pctobnew_3','pctobnew_4','pctobnew_5','pctobnew_6'], axis=1)
    #print(x_train.dtypes)
    #print(x_test.dtypes)
    y_train = x_train['class']
    
    #x_train = fet(x_train)

    x_train['paper_vitality'] = 0.473 * x_train['paper_vitality']  
    x_train['growth_stage'] = 0.292 * x_train['growth_stage'] 
    x_train['citation_vitality'] = 0.1 * x_train['citation_vitality'] 

    x_train[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new','citation_vitality','paper_vitality','growth_stage']] = minmax_scale(x_train[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new','citation_vitality','paper_vitality','growth_stage']])
    x_train = x_train.drop(['n','cited_by_clin','diffpctob','diffpct','cluster','year','class',"pctisnew_7",'pprev','pct_in_biggest_anc','pct_of_biggest_anc','rcr_low'], axis = 1)#$%
    #x_train = x_train.drop(['pct_of_biggest_anc_new','pct_of_secbiggest_anc_new','animal','pct_is_newish'], axis = 1)
    x_train = x_train.drop(['pct_is_newish','pct_of_biggest_anc_new','paper_vitality','citation_vitality','growth_stage'], axis = 1)
    #x_train['counts'] = x_train['counts'] - x_train['counts'].mean()
    #x_train['counts'] = x_train['counts']/x_train['counts'].std()

    ##dropping highly relevant features from direct citation training data via coefficient matrix
    #x_train = x_train.drop(['biggest_anc','human','pct_of_biggest_anc_new','n_biggest_anc','pctisnew_1','pctisnew_7','pctisnew_3','pctisnew_4','pctisnew_5','pctobnew_7'], axis=1)


    #print(x_train.dtypes)

    if year == 2017:
        print(x_train.dtypes)
    y_train = np.array(y_train)
    x_train = x_train.values

    data = pd.read_csv("with250_previous_years_"+str(year)+".csv")
    ####remove here
    data = cutoff(data)


    column_means = data.mean()
    data = data.fillna(column_means)

    x_test = data
    #x_test = x_test.drop(['pctobnew_1','pctobnew_2','pctobnew_3','pctobnew_4','pctobnew_5','pctobnew_6'], axis=1)
    model = LogisticRegression(random_state=0, C = 0.40).fit(x_train,y_train) ##logistioc
    

    pred = model.predict(x_train)
    #print(pred[:50])


    print("f1 score ",f1_score(y_train, pred , average="macro"))

    temp = x_test
    
    #x_test = fet(x_test)
    x_test['counts'] = np.log10(x_test['counts']+1)
    x_test = newfet_test(x_test,year)


    
    x_test['paper_vitality'] = 0.473 * x_test['paper_vitality']  
    x_test['growth_stage'] = 0.292 * x_test['growth_stage'] 
    x_test['citation_vitality'] = 0.1 * x_test['citation_vitality'] 

    x_test[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new','citation_vitality','paper_vitality','growth_stage']] = minmax_scale(x_test[['wrcr','rcr_hi','rcr_mid','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc','pct_of_biggest_anc_new','citation_vitality','paper_vitality','growth_stage']])
    x_test = x_test.drop(['n','pct_of_biggest_anc_newish','rcr_low','pct_of_biggest_anc','pct_in_secbiggest_anc','pct_of_secbiggest_anc','pct_of_secbiggest_anc_newish'], axis = 1)
    x_test = x_test.drop(['prediction','year','cluster','biggest_anc','pct_in_biggest_anc','secbiggest_anc','cited_by_clin','pprev'], axis=1) #$%
    #x_test = x_test.drop(['pct_of_biggest_anc_new','pct_of_secbiggest_anc_new','animal','pct_is_newish'], axis = 1)
    x_test = x_test.drop(['pct_is_newish','pct_of_biggest_anc_new','paper_vitality','citation_vitality','growth_stage'], axis = 1)
    #x_test['counts'] = x_test['counts'] - x_test['counts'].mean()
    #x_test['counts'] = x_test['counts']/x_test['counts'].std()

    

    #####add here

    #print(x_test.dtypes)
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



    print(gold)

    isgold = []

    for i in range(len(temp)):
        isgold.append("FALSE")


    for ind in gold.index:
        index = temp.index
        condition = temp["cluster"] == gold['cluster'][ind]
        apples_indices = index[condition]
        #print(apples_indices)
        isgold[apples_indices[0]] = "TRUE"



    temp['isgold'] = isgold
    ##remove here
    #temp.to_csv("xxxx_"+str(year)+".csv", index = False)

    x1 = temp[temp.isgold == "FALSE"]
    x2 = temp[temp.isgold == "TRUE"]
    y1 = np.random.rand(len(x1))
    y2 = np.random.rand(len(x2))
    plt.scatter(x1['pred_log'], y1, s=1, c='b', alpha=0.5)
    plt.scatter(x2['pred_log'], y2, s=1, c='r', alpha=0.5)
    plt.xlabel("prediction log")
    #plt.show()

    return calc_recall(year,temp,th)



y = 1984
sum = 0
Fp = 0
FN = 0
ex = 0
ss = 0
correct_pos = 0
tot_pos = 0
positive_training = 0
pos = []
while y <= 2017:
    a,fp,fn,EX,e,f, g = train(y,0.74)
    sum += a
    Fp += fp
    FN += fn
    ex += EX
    positive_training += g
    print("for year ",y," CSI score ",a/(a+fp+fn))
    correct_pos += e
    tot_pos += f
    pos.append(f)
    ss += (a/(a+fp+fn))
    y += 1
    
print("sum ", sum," percent ", sum/475)
print("Fp ", Fp," percent ", Fp/34)
print("FN ", FN)
print("EX ", ex)
print("ss ", ss/34)
print("correct positive ", correct_pos)
print("total pos ", tot_pos)
print("positive_training ", positive_training)
print(pos)


'''th = 0.00
Th,F,T = [],[],[]
while th <= 0.99:
    y = 1984
    sum = 0
    Fp = 0
    while y <= 2017:
        a,fp,fn,EX = train(y,th)
        sum += a
        Fp += fp
        y += 1
    Th.append(th)
    F.append(Fp)
    T.append(sum)
    th += 0.01
plt.scatter(Th, T, s = 5)
plt.ylabel('recall')
plt.xlabel('threshold')
plt.show()
'''
'''plt.scatter(Th, F, s = 5)
plt.ylabel('false_positive')
plt.xlabel('threshold')
plt.show()'''

'''
SVM: C = 1.00, th = 0.75
recall: 386, FP: 21304

SVM: C = 1.00, th = 0.73
recall: 388, FP: 23421

SVM: C = 1.00, th = 0.70
recall: 394, FP: 26870
'''

'''
LR: C = 0.40, th = 0.70
recall: 380, FP: 20053

LR: C = 0.80, th = 0.70
recall: 383, FP: 23726

LR: C = 1.00, th = 0.70
recall: 387, FP: 25142

LR: C = 0.40, th = 0.65
recall: 392, FP: 25806

LR: C = 0.60, th = 0.65
recall: 396, FP: 27768

LR: C = 0.50, th = 0.65
recall: 394, FP: 26847

lr: C = 0.31, th = 0.62
features:
pct_is_new                   float64
n_biggest_anc                float64
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
is_clinical                  float64
is_research_article          float64
nih                          float64
rage                         float64
prev                         float64
ppprev                       float64
recall: 398, fp: ~967(24869)
another good one: C= 0.50, th = 0.75, tp: 358, 483
after cutofff: n < 100: C = 0.35, y = 0.73, sum = 361, FP = 6657
n < 100: C = 0.38, y = 0.74, sum = 360, FP = 6278

another good one:
using fet data: C: 0.50, th: 0.70
recall: 390, fp: 726 (total = 16901)
after cutofff: n < 100: C = 0.30, y = 0.73, sum = 359, FP = 6779
'''

'''data = pd.read_csv("extreme_training.csv")
sys.stdout = open("correlation.txt", "w")
data = data.drop(['n','cited_by_clin','diffpctob','diffpct','cluster','year',"pctisnew_7",'pprev','pct_in_biggest_anc','pct_of_biggest_anc','rcr_low'], axis = 1)#$%
corr = data.corr(method='kendall')
print(corr)
corr_1 = corr.iloc[:,0:4]
print("printing correlation")
print(corr_1)
corr_2 = corr.iloc[:,4:7]
print("printing correlation")
print(corr_2)
corr_3 = corr.iloc[:,7:11]
print("printing correlation")
print(corr_3)
corr_4 = corr.iloc[:,11:14]
print("printing correlation")
print(corr_4)
corr_5 = corr.iloc[:,14:17]
print("printing correlation")
print(corr_5)
corr_6 = corr.iloc[:,17:20]
print("printing correlation")
print(corr_6)
corr_7 = corr.iloc[:,20:24]
print("printing correlation")
print(corr_7)
corr_8 = corr.iloc[:,24:]
print("printing correlation")
print(corr_8)
print("kendall's correlation")'''