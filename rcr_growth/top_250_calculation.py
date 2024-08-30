import pandas as pd
import numpy as np

'''top = pd.read_csv("top_250_journal.csv")
x = top['year'].unique()
#print(top.head(5))
for i in x:
    data = top[top.year == i]
    print("len ", len(data))
    data = data[data.top250 == True]
    print("len ", len(data))
    data.to_csv("top_"+str(i)+".csv", index = False)

'''


y = 1984
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
'''data = pd.read_csv("leiden_cluster_table_"+str(y)+".tsv", sep = "\t")
data = data[data.id == 6732201]
print(data)
'''