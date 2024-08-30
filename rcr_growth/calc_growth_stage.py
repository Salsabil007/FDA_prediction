import pandas as pd
import numpy as np

y = 1984


'''df = pd.read_csv("hicite_apt_2021.tsv", sep = "\t")
print(df.dtypes)
value_counts = df['year'].value_counts()
df_val_counts = pd.DataFrame(value_counts)
df_value_counts_reset = df_val_counts.reset_index()
df_value_counts_reset.columns = ['year', 'counts']
df_value_counts_reset.to_csv("year_value.csv", index = False)'''


data = pd.read_csv("leiden_cluster_table_"+str(y)+".tsv", sep = "\t")
data = data.rename(columns = {'id':'pmid'})
data = data.drop_duplicates()
data = data.sort_values(by=['cl'])
#print(data['cl'].unique())
#print(data[data.cl == 118343], " ghgdhgghd")

d = pd.read_csv("hicite_apt_2021.tsv", sep = "\t")
df = pd.DataFrame()
df['pmid'] = d['pmid']
df['year'] = d['year']
df = df.drop_duplicates()
#print(len(df))

result = pd.merge(data,df, on = 'pmid', how = 'inner')
result = result.drop(['pmid'], axis = 1)
print(result.dtypes)
result = result.groupby(["cl", "year"]).size().reset_index(name="sum")

print(len(result))

y_val = pd.read_csv("year_value.csv")
result = pd.merge(result,y_val,on=['year'], how = 'inner')
print(len(result))

cluster = []
peak_year = []
xx = result['cl'].unique()
#print(xx)
for i in xx:
    t = result[result.cl == i]
    val = t['sum']/t['counts']
    t['div'] = val
    t = t.sort_values(by=['div'], ascending=False)
    cluster.append(i)
    t = t.head(1)
    t = t.to_numpy()
    peak_year.append(int(t[0][1]))
growth_yr = pd.DataFrame()
growth_yr['cluster'] = cluster
growth_yr['peak_year'] = peak_year
growth_yr = growth_yr.sort_values(by=['cluster'])

growth_stage = 1/(1+y-growth_yr['peak_year'])
growth_yr['growth_stage'] = growth_stage
growth_yr.to_csv("growth_year_"+str(y)+".csv", index= False)



'''
##draft testing section
d= {'col1': [1, 2, 1,1,2], 'col2': ['a', 'b','a','a','c']}
df = pd.DataFrame(data=d)
print(df)
df = df.groupby(["col1", "col2"]).size().reset_index(name="sum")
print(df)'''