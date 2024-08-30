import pandas as pd
import numpy as np

'''
##step 1: For each year, just keep those pmids that were published in or before that year. Separate files for easy handling. This part needs to be run just once at the beginning.
top = pd.read_csv("top_250_journal.csv")
x = top['year'].unique()
#print(top.head(5))
for i in x:
    data = top[top.year <= i]
    print("len ", len(data))
    data = data[data.top250 == True]
    print("len ", len(data))
    data.to_csv("top_"+str(i)+".csv", index = False)

'''


'''
##step 2: Top 250 journal count for each year and each cluster. 
y = 1979
while y <= 2020:
    df = pd.read_csv("top_"+str(y)+".csv")
    print("len before ", len(df))
    df = df.rename(columns={'pmid': 'id'})
    print(df.dtypes)
    print("len after ", len(df))
    data = pd.read_csv("leiden_cluster_table_"+str(y)+".tsv", sep = "\t")
    result = pd.merge(df, data, how = "inner", on=["id"])
    print("len of result ", len(result))
    result.to_csv("xxx.csv", index = False)
    
    value_counts = result['cl'].value_counts()
    df_val_counts = pd.DataFrame(value_counts)
    df_value_counts_reset = df_val_counts.reset_index()
    df_value_counts_reset.columns = ['unique_cl', 'counts']
    #print (df_value_counts_reset)
    df_value_counts_reset.to_csv("journal_count_"+str(y)+".csv", index = False)
    y += 1

'''

'''##step 3: merge the journal count with our actual feature file.
y = 1979
while y <= 2018:
    data = pd.read_csv("feature_"+str(y)+".csv")
    jr = pd.read_csv("journal_count_"+str(y)+".csv")
    #print(jr.dtypes)
    jr = jr.rename(columns={'unique_cl': 'cluster'})
    tt = jr
    print("len before ", len(tt))
    result = pd.merge(data,tt,on=['cluster'],how= 'left')
    print("len of data ", len(data))
    print("len of result ", len(result))
    result['counts'] = result['counts'].fillna(0)
    result = result.sort_values(by=['cluster'], ignore_index=True)
    result.to_csv("with250_feature_"+str(y)+".csv", index = False)
    y += 1


##During training, we need to take log of the journal counts.
'''

'''
#for test data
y = 2020
data = pd.read_csv("test_prev_year.csv")
jr = pd.read_csv("journal_count_"+str(y)+".csv")
jr = jr.rename(columns={'unique_cl': 'cluster'})
tt = jr
print("len before ", len(tt))
result = pd.merge(data,tt,on=['cluster'],how= 'left')
print("len of data ", len(data))
print("len of result ", len(result))
result['counts'] = result['counts'].fillna(0)
result = result.sort_values(by=['cluster'], ignore_index=True)
result.to_csv("with250_test.csv", index = False)
   
'''

##for training data:
data = pd.read_csv("training_prev_year2.csv")
counts = []
for ind in data.index:
    y = data['year'][ind]
    cl = data['cluster'][ind]
    tt = pd.read_csv("journal_count_"+str(y)+".csv")
    tt = tt[tt.unique_cl == cl]
    if len(tt) == 0:
        counts.append(0)
        continue
    tt = tt.to_numpy()
    counts.append(tt[0][1])
data['counts'] = counts
data.to_csv("with250_training.csv", index= False)