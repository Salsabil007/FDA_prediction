from operator import index
import pandas as pd
import numpy as np

def cal_citvit(y):
    ##part 1
    occ_y = pd.read_csv("occ_year_"+str(y)+".csv")
    #occ_y = occ_y.drop(['citing','citing_year'], axis = 1)
    occ_y = occ_y.drop(['referenced_year','citing'], axis = 1)
    occ_y['citing_year'] = y - occ_y['citing_year']
    occ_y['citing_year'] = 1 / (1 + occ_y['citing_year'])

    #tmp = occ_y[occ_y.referenced == 34141351]
    #print(tmp['citing_year'].sum())
    #print(occ_y.tail(5))
    occ_y = occ_y.groupby(['referenced']).sum().reset_index()
    print(occ_y.tail(5))
    print(occ_y.dtypes)
    occ_y.to_csv("ref_age_sum_"+str(y)+".csv", index = False)
    print(len(occ_y))


    ##part 2
    data = pd.read_csv("leiden_cluster_table_"+str(y)+".tsv", sep = "\t")
    data = data.rename(columns = {'id':'pmid'})
    data = data.drop_duplicates()
    data = data.sort_values(by=['cl'])

    d = pd.read_csv("hicite_apt_"+str(y)+".tsv", sep = "\t")
    df = pd.DataFrame()
    df['pmid'] = d['pmid']
    df['citation_count'] = d['citation_count']
    df = df.drop_duplicates()
    #print(len(df))

    result = pd.merge(data,df, on = 'pmid', how = 'inner')
    print(len(result))
    print(result.dtypes)

    occ = pd.read_csv("ref_age_sum_"+str(y)+".csv")
    occ = occ.rename(columns = {'referenced':'pmid'})

    result2 = pd.merge(result,occ, on='pmid', how = 'inner')
    print(len(result))
    print(len(result2))

    result2['citing_year'] = result2['citing_year']/result2['citation_count']
    print(result2.dtypes) 
    result2 = result2.drop(['pmid','citation_count'], axis = 1)
    result2 = result2.groupby(['cl']).sum().reset_index()
    result2.to_csv("cit_vit_"+str(y)+".csv", index = False)


y = 2020
while(y<=2020):
    cal_citvit(y)
    y += 1

##we need to divide it by n and power (1/4). We will do that during merging with the features_prediction_1999.csv file.
 

