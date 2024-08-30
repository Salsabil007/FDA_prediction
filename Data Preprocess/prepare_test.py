import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, minmax_scale
from sklearn.utils import shuffle

data = pd.read_csv("2020_overlap.csv")
#print(data.isnull().sum())
#print(data.dtypes)
#data = shuffle(data)
#data = shuffle(data)
#data = shuffle(data)
#data = shuffle(data)
#data = shuffle(data)
#data = shuffle(data)
#data = shuffle(data)
#data = shuffle(data)
#data = shuffle(data)
#data = shuffle(data)
#data = data.drop(['prediction','intersect','pct_covid'], axis=1)

'''
lbl = LabelEncoder()
data['year'] = lbl.fit_transform(data['year'].values)
data['n'] = lbl.fit_transform(data['n'].values)
data['n_biggest_anc'] = lbl.fit_transform(data['n_biggest_anc'].values)
data['n_secbiggest_anc'] = lbl.fit_transform(data['n_secbiggest_anc'].values)
data['n_clusts_90_anc'] = lbl.fit_transform(data['n_clusts_90_anc'].values)
'''
data[['wrcr','rcr_mid','rcr_hi','rage']] = minmax_scale(data[['wrcr','rcr_mid','rcr_hi','rage']])
data[['year','n','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc']] = minmax_scale(data[['year','n','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc']])

x = data.drop(['cluster','n','pct_is_new','pct_of_biggest_anc_newish','rcr_hi','biggest_anc','secbiggest_anc'], axis = 1)
print(x.dtypes)
print(x.describe())
x.to_csv("test_final.csv", index = False)
