import pandas as pd
import numpy as np
'''
data = pd.read_csv("training_prev_year2.csv")


jr = pd.read_csv("journal_count_1984.csv")
print(jr.dtypes)
jr['year']=1984
jr = jr.rename(columns={'unique_cl': 'cluster'})


tt = jr
print("len before ", len(tt))
y =1985
while(y<=2017):
    jr = pd.read_csv("journal_count_"+str(y)+".csv")
    #print(jr.dtypes)
    jr['year']=y
    jr = jr.rename(columns={'unique_cl': 'cluster'})
    #print("len middle ", len(jr))
    tt = tt.append(jr, ignore_index=True)
    #print("len after ", len(tt))
    y += 1

result = pd.merge(data,tt,on=['year','cluster'],how= 'left')
print("len of data ", len(data))
print("len of result ", len(result))
result['counts'] = result['counts'].fillna(0)
result.to_csv("with250_training_prev_year2.csv", index = False)'''


'''data = pd.read_csv("training_prev_year3.csv")


jr = pd.read_csv("journal_count_1984.csv")
print(jr.dtypes)
jr['year']=1984
jr = jr.rename(columns={'unique_cl': 'cluster'})


tt = jr
print("len before ", len(tt))
y =1985
while(y<=2017):
    jr = pd.read_csv("journal_count_"+str(y)+".csv")
    #print(jr.dtypes)
    jr['year']=y
    jr = jr.rename(columns={'unique_cl': 'cluster'})
    #print("len middle ", len(jr))
    tt = tt.append(jr, ignore_index=True)
    #print("len after ", len(tt))
    y += 1

result = pd.merge(data,tt,on=['year','cluster'],how= 'left')
print("len of data ", len(data))
print("len of result ", len(result))
result['counts'] = result['counts'].fillna(0)
result.to_csv("with250_training_prev_year3.csv", index = False)'''



'''data = pd.read_csv("test_prev_year2.csv")


jr = pd.read_csv("journal_count_2020.csv")
print(jr.dtypes)
jr['year']=2020
jr = jr.rename(columns={'unique_cl': 'cluster'})


tt = jr
print("len before ", len(tt))

result = pd.merge(data,tt,on=['year','cluster'],how= 'left')
print("len of data ", len(data))
print("len of result ", len(result))
result['counts'] = result['counts'].fillna(0)
result.to_csv("with250_test_prev_year2.csv", index = False)'''


y = 1984
while y <= 2017:
    data = pd.read_csv("previous_years_"+str(y)+".csv")
    jr = pd.read_csv("journal_count_"+str(y)+".csv")
    #print(jr.dtypes)
    jr = jr.rename(columns={'unique_cl': 'cluster'})
    tt = jr
    print("len before ", len(tt))
    result = pd.merge(data,tt,on=['cluster'],how= 'left')
    print("len of data ", len(data))
    print("len of result ", len(result))
    result['counts'] = result['counts'].fillna(0)
    result.to_csv("with250_previous_years_"+str(y)+".csv", index = False)
    y += 1