##in this file, we are trying to merge the rcr of papers for each year
import pandas as pd
import numpy as np

'''data = pd.read_csv("hicite_apt_2021.tsv", sep="\t")
print(len(data))
d = pd.DataFrame()
d['pmid'] = data['pmid']
d['rcr_2021'] = data['relative_citation_ratio']
print(d.dtypes)
print(len(d))
d = d.drop_duplicates()
print(len(d))
d.to_csv("rcr_2021.csv", index = False)'''

'''data = {'Name':[1,2,3,4,5,6],
        'Age1':[.5,.6,.7,.8,.9,.3]}
  
# Create DataFrame
df1 = pd.DataFrame(data)

data = {'Name':[1,2,5,6,9],
        'Age2':[.1,.2,.33,.4,.55]}
  
# Create DataFrame
df2 = pd.DataFrame(data)

result = pd.merge(df1, df2, how="left", on=["Name"])
print(result)'''

'''data1 = pd.read_csv("rcr_2021.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2020.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))
'''
'''
data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2019.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))

'''

'''
data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2018.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))
'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2017.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))
'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2016.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2015.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2014.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))
'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2013.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))
'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2012.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))
'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2011.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2010.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))
'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2009.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2008.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2007.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2006.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))'''

'''data1 = pd.read_csv("rcr_merged.csv")
print(len(data1))

data2 = pd.read_csv("rcr_2005.csv")
print(len(data2))

result = pd.merge(data1, data2, how="left", on=["pmid"])
result.to_csv("rcr_merged.csv", index = False)
print(len(result))'''