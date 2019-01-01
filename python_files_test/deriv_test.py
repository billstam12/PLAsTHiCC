
import numpy as np
import pandas as pd
import read_test

info = read_test.init_reading()


from multiprocessing import Pool
import multiprocessing as mp

def applyParallel_1(dfGrouped, func):
    p = Pool(mp.cpu_count())
    ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list, axis=1)

def derivative(series):
    #mn_pos = np.mean(series[(series[["flux"]] >= 0)].flux)
    #mn_neg =np.mean(series[(series[["flux"]] < 0)].flux)
    mn = np.mean(series.flux)
    series = series[(series["flux"]>mn)]
    fl_d = np.diff(series["flux"])
    tm_d = np.diff(series["mjd"])
    deriv = fl_d/tm_d
    count1 = 0
    for i in range(len(deriv)-1):
        if(deriv[i]*deriv[i+1]<0):
            count1+=1
    return count1

from tqdm import tqdm

for i_c, (end,start) in enumerate(tqdm(read_test.get_chunks(info, 100000))):

    df = (read_test.read_object_by_index_range(info,start,end))

    #Normalize flux
    #df['flux'] = df['flux']/13675792.0


    counts = df.groupby(["object_id", "passband"]).apply(derivative)
    counts = counts.unstack(level=-1)

    new_columns = [
        "no_of_changes" + str(i)  for i in range(6)
    ]
    counts.columns = new_columns


    if(i_c == 0):
        counts.to_csv("test/neg_flux/derivs_test.csv", header = True, index = True, mode='a')
    else:
        counts.to_csv("test/neg_flux/derivs_test.csv", header = False, index = True, mode='a')

      