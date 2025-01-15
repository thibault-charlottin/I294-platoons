import pandas as pd
import numpy as np
import src.test_CF as test_CF
    


def define_veh_string(dtw_data,trajs_data, max_dtw):
    #allows for the cut of cut-in cenarios
    trajs_data = determine_CF(trajsdf,DTW_df,max_dtw)
    leaders = np.array(dtw_data['leader'])
    followers = np.array(dtw_data['id'])
    dtw_value = np.array(dtw_data['dtw'])
    string_list = []
    string = []
    for k in range(1,len(leaders)):
        if leaders[k]==followers[k-1] and dtw_value[k] < max_dtw:
            string.append(followers[k-1])
        else : 
            string_list.append(string)
            string = []
    return string_list

