


import numpy as np
import pandas as pd
import read_test
info = read_test.init_reading()

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from multiprocessing import Pool
import multiprocessing as mp

def applyParallel_1(dfGrouped, func):
    p = Pool(mp.cpu_count())
    ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list, axis=1)


def dtw_query(series):
    study_df = series[["flux","passband"]]
    dist = []
    for i in range(5):
        for j in range(i+1,6):
            dist.append(fastdtw(study_df[study_df.passband==i],study_df[study_df.passband==j], dist = euclidean)[0])
    return pd.DataFrame(dist)

from tqdm import tqdm

for i_c, (end,start) in enumerate(tqdm(read_test.get_chunks(info, 100000))):
    if(i_c <30):
       continue
    else:
    	df = (read_test.read_object_by_index_range(info,start,end))

    	#Normalize flux
    	#df['flux'] = df['flux']/13675792.0


    	dtw_test = applyParallel_1(df.groupby(["object_id"]),dtw_query)
    	new_columns = [
        	"DTW" + str(k) + "-" + str(l) for k in range(5) for l in range(k+1,6)
    	]
    	dtw_test = dtw_test.T
    	dtw_test.columns = new_columns

    	if(i_c == 0):
        	dtw_test.to_csv("test/neg_flux/dtw_test2.csv", header = True, index = False, mode='a')
    	else:
        	dtw_test.to_csv("test/neg_flux/dtw_test2.csv", header = False, index = False, mode='a')
