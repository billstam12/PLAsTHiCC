import numpy as np
import pandas as pd
import matplotlib
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
import read_test
info = read_test.init_reading()


aggs_all = {
        'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        'detected': ['mean']
}
    
from itertools import islice

def sxedio5(series):
    threshold = 120
    diffs = np.diff(series.mjd)
    diff_ids = (np.where(diffs>threshold))
    print(diff_ids)
    """
    if(len(diff_ids[0])==2):
        series1 = series[:diff_ids[0][0]]
        series2 = series[diff_ids[0][0]:diff_ids[0][1]]
        series3 = series[diff_ids[0][1]:]
        
        agg1 = series1.agg(aggs_all)
        agg2 = series2.agg(aggs_all)
        agg3 = series3.agg(aggs_all)

        aggs = pd.concat([agg1, agg2, agg3], axis=1)
    elif(len(diff_ids[0])==3):
        series1 = series[:diff_ids[0][0]]
        series2 = series[diff_ids[0][0]:diff_ids[0][1]]
        series3 = series[diff_ids[0][1]:diff_ids[0][2]]
        series4 = series[diff_ids[0][2]:]

        agg1 = series1.agg(aggs_all)
        agg2 = series2.agg(aggs_all)
        agg3 = series3.agg(aggs_all)
        agg4 = series4.agg(aggs_all)

        aggs = pd.concat([agg1, agg2, agg3, agg4], axis=1)
    else:
        mjd_diff = np.max(series.mjd) - np.min(series.mjd)
        mjd_diff_div =int(np.ceil(mjd_diff/4))
        #print(mjd_diff, mjd_diff_div)
        series1 = series[:mjd_diff_div]
        series2 = series[mjd_diff_div:2*mjd_diff_div]
        series3 = series[2*mjd_diff_div:3*mjd_diff_div]
        series4 = series[3*mjd_diff_div:]
       
        
        agg1 = series1.agg(aggs_all)
        agg2 = series2.agg(aggs_all)
        agg3 = series3.agg(aggs_all)

        aggs = pd.concat([agg1, agg2, agg3], axis=1)
    if(aggs.shape == (6,9)):
        agg1[:] = 0
        #print(agg1)
        aggs = pd.concat([aggs,agg1],axis=1)
    return aggs   
    """
    
    
    
for i_c, (end,start) in enumerate(read_test.get_chunks(info, 100000)):
    
    df = (read_test.read_object_by_index_range(info,start,end))

    new_feats = df.groupby("object_id").apply(sxedio5)
    new_feats = new_feats.unstack(level=-1)
    new_columns = [
                "new_feats" + str(i)  for i in range(72)
            ]
    new_feats.columns = new_columns
 

        
