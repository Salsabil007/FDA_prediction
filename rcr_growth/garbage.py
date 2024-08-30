import pandas as pd
import numpy as np

def rcr_cnt(yr,cl):
    tab = pd.read_csv("leiden_cluster_table_"+str(yr)+".tsv", sep = "\t")
    tab = tab[tab.cl == cl]
    #print(len(tab))
    tab = tab.sort_values(by = ["id"])
    #tab = tab.head(10)
    pid = tab["id"].unique()

    cnt = 0
    for i in pid:
        x = pd.read_csv("rcr_"+str(yr)+".csv")
        x = x[x.pmid == i]
        x = x.to_numpy()
        cur_rcr = x[0][1]
        if np.isnan(cur_rcr):
            cur_rcr = 0 
        prev_rcr = 0
        j = 1976
        while j <= yr:
            dd = pd.read_csv("rcr_"+str(j)+".csv")
            dd = dd[dd.pmid == i]
            if len(dd) > 0:
                dd = dd.to_numpy()
                if (np.isnan(dd[0][1]) or dd[0][1] == 0):
                    j += 1
                    continue
                prev_rcr = dd[0][1]
                print("prev_rcr ", prev_rcr," year ",j," pid ",i," curr ",cur_rcr)
                break
            else:
                j += 1
        if (cur_rcr >= 3.00 and cur_rcr >= prev_rcr * 3.00):
            cnt += 1
    return cnt/len(tab)

data = pd.read_csv("training_selected.csv")
#data = data.tail(10)
print(data)

cnt = []
for ind in data.index:
    yr = data["year"][ind]
    cl = data["cluster"][ind]
    cnt.append(rcr_cnt(yr,cl))
data["rcr_growth"] = cnt
data.to_csv("training_with_rcr.csv", index = False)