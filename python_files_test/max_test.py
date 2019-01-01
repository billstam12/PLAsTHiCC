import numpy as np
import pandas as pd
import read_test
import gc; gc.enable()

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from multiprocessing import Pool
import multiprocessing as mp

def applyParallel_1(dfGrouped, func):
    p = Pool(mp.cpu_count())
    ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list, axis=1)

def find_maxs_1(series):
    count = 0
    series = np.array(series)
    for i in range(1,len(series)-1):
        if (series[i] > series[i-1]) & (series[i] > series[i+1]):
            if(series[i]>np.mean(series)):
                count +=1
    return count

from tqdm import tqdm

remain_df = None 
for i_c, df in enumerate(pd.read_csv('all/test_set.csv', chunksize=5000000, iterator=True)):
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        unique_ids = np.unique(df['object_id'])
        
        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()
        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])]
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)
        # Create remaining samples df
        remain_df = new_remain_df
        
        aggs = {'flux': [find_maxs_1] }
        max_test = df.groupby(['object_id','passband']).agg(aggs)
        max_test = max_test.unstack(level=-1)
        new_columns = [
            "no_of_max_" + str(i)  for i in range(6)
        ]
    
        max_test.columns = new_columns
        
        
        if i_c == 0:
            max_test.to_csv("test/neg_flux/max_test2.csv", header=True, index=True, mode='a')
        else:
            max_test.to_csv("test/neg_flux/max_test2.csv", header=False, index=True, mode='a')
    
        del max_test
        gc.collect()

remain_df.to_csv("test/neg_flux/max_remain.csv", index=True)       