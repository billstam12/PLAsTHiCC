
import numpy as np
import pandas as pd
import read_test
info = read_test.init_reading()
from tqdm import tqdm



from multiprocessing import Pool
import multiprocessing as mp


def test(series, threshold):
    series.reset_index(inplace=True)
    mean = series.flux.mean()
    
    max_flux = series.flux.idxmax()
    series1 = (series[:max_flux])
    series2 = (series[max_flux:])
    i = -1
    try:
        while(np.array(series1.flux)[i] > np.array(series1.flux)[i-1]-(threshold*mean)):
            i-=1
        mjd1 = np.array(series1.mjd)[i]

        i = 0

        while(np.array(series2.flux)[i] > np.array(series2.flux)[i+1]-(threshold*mean)):
            i+=1

        mjd2 = np.array(series2.mjd)[i]
        mjd = mjd2 - mjd1
    except:
        mjd = 0
    return mjd


for i_c, (end,start) in enumerate(tqdm(read_test.get_chunks(info, 100000))):
 
    df = (read_test.read_object_by_index_range(info,start,end))
    
    s = df.groupby("object_id").apply(test, 0.1)
    s = pd.DataFrame(s)
    s.columns = ["mjd_my_diff"]
    s.to_csv("test/neg_flux/mjd_diffs.csv")
    
    if(i_c == 0):
        s.to_csv("test/neg_flux/mjd_diffs.csv", header = True, index = False, mode='a')
    else:
        s.to_csv("test/neg_flux/mjd_diffs.csv", header = False, index = False, mode='a')