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


data = pd.read_csv("training.csv")
print(data.dtypes)
value_counts = data['class'].value_counts()
print("value_counts ", value_counts)
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
print(data.head(10))
#print(data.head(10))

#print(data.isnull().sum())
#print(data.describe())
data.dropna(axis=0, inplace= True)


print(data.head(5))

##data scaling and encoding
#lbl = LabelEncoder()
#data['year'] = lbl.fit_transform(data['year'].values)
#data['n'] = lbl.fit_transform(data['n'].values)
#data['n_biggest_anc'] = lbl.fit_transform(data['n_biggest_anc'].values)
#data['n_secbiggest_anc'] = lbl.fit_transform(data['n_secbiggest_anc'].values)
#data['n_clusts_90_anc'] = lbl.fit_transform(data['n_clusts_90_anc'].values)
data[['wrcr','rcr_mid','rcr_hi','rage']] = minmax_scale(data[['wrcr','rcr_mid','rcr_hi','rage']])
data[['year','n','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc']] = minmax_scale(data[['year','n','n_biggest_anc','n_secbiggest_anc','n_clusts_90_anc']])

print(data.head(5))

'''
### This section is for checking correlation among features
sys.stdout = open("correlation.txt", "w")
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


#dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
#plt.show()
'''



data['class'] = data['class'].astype(int)
y = data['class']
y = np.array(y)
#x = data.drop(['class'], axis=1)
##drop highly correlated features
x = data.drop(['cluster','n','pct_is_new','pct_of_biggest_anc_newish','rcr_hi','pct_dusted_ccn','pct_dusted_rmcl','biggest_anc','secbiggest_anc'], axis = 1)
print(x.dtypes)
print(x.describe())
x.to_csv("training_final.csv", index = False)

'''
###building the model
x = x.values
print("shape of input ", x.shape)
print("shape of y ", y.shape)
model = Sequential()
model.add(Dense(1, activation='sigmoid',input_dim=x.shape[1]))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
history = model.fit(x, y,epochs=130, batch_size=8, verbose=1)
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
plt.show()

eval = model.evaluate(x,y, verbose = 1)
print("model evaluation ",eval)

pred = model.predict(x)
print(pred[:50])


predy = np.where(pred > 0.5, 1,0)
print(predy[:5])
print(f1_score(y, predy , average="macro"))


test_x = pd.read_csv("test_final.csv")
test_x = shuffle(test_x)
test_x = shuffle(test_x)
test_x = shuffle(test_x)
test_x = shuffle(test_x)
test_x = shuffle(test_x)

test_x = test_x.values
pred_test = model.predict(test_x)
print(pred_test[:200])
p = np.where(pred_test >= 0.5, 1,0)
print(p[:200])
'''






