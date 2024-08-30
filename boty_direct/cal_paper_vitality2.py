import pandas as pd
import numpy as np


def cal_vit(y):
    data = pd.read_csv("leiden_cluster_table_"+str(y)+".tsv", sep = "\t")
    data = data.rename(columns = {'id':'pmid'})
    data = data.drop_duplicates()
    data = data.sort_values(by=['cl'])

    d = pd.read_csv("hicite_apt_2021.tsv", sep = "\t")
    df = pd.DataFrame()
    df['pmid'] = d['pmid']
    df['year'] = d['year']
    df = df.drop_duplicates()
    #print(len(df))

    result = pd.merge(data,df, on = 'pmid', how = 'inner')
    print(len(result))
    print(len(data))

    age = y - result['year']
    result['age'] = age
    result['val'] = 1/(result['age'] + 1)
    result = result.groupby(["cl"]).sum().reset_index()
    result = result.sort_values(by=['cl'])
    result.to_csv("paper_vit2_"+str(y)+".csv", index = False)

y = 2020
while(y<=2020):
    cal_vit(y)
    y += 1


### when combining with each cluster of each year, we need to devide the val by n.