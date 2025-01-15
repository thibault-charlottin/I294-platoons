import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from dtaidistance.preprocessing import differencing
from tqdm.notebook import trange
import seaborn as sns
sns.set_theme()

def test_car_following_by_time(data,id1,id2,tau,window,step):
    x1 = data[data['ID']==id1]
    x2 = data[(data['ID']==id2)]
    T = x2[x2['leader']==id1]['time']
    DTW = []
    time = []
    k = min(T)
    while k<max(T):
        tmin_lead = int(k)
        tmax_lead = min(int(k+(window)),max(T))
        tmin_follow = k+tau
        tmax_follow = k+window+tau
        try : 
            X1 = differencing(np.array(x1[(x1['time']<=tmax_lead)&(x1['time']>=tmin_lead)]['r']), smooth=0.1)
            X2 = differencing(np.array(x2[(x2['time']<=tmax_follow)&(x2['time']>=tmin_follow)]['r']), smooth =0.1)
            distance, paths = dtw.warping_paths(X1, X2)
            DTW.append(distance)
        except Exception :
            DTW.append(np.nan)
        time.append(k)
        k=k + step
    return time,DTW


def prepareTaudf(data,dfTau,leader,follower):
    leader = float(leader)
    follower = float(follower)
    maxtimelists = []
    dfTau = dfTau[(dfTau['leader']==leader)&(dfTau['id']==follower)]
    Tau = [k for k in  dfTau['Tau']]
    maxtimelists.append(max(data[data['ID']==follower]['time']))
    return {'time':maxtimelists,'tau':Tau}


def test_car_following_by_time_variable_Tau(data,id1,id2,tau_dic,window,step):
    x1 = data[data['ID']==id1]
    x2 = data[(data['ID']==id2)]
    T = x2[x2['leader']==id1]['time']
    DTW = []
    time = []
    k = min(T)
    tau = tau_dic['tau'][0]
    timeTau = tau_dic['time'][0]
    iterate = 0
    while k<max(T):
        if (T.reset_index(drop=True) > timeTau).all():
            iterate+=1
            tau = tau_dic['tau'][iterate]
            timeTau = tau_dic['time'][iterate]
        tmin_lead = int(k)
        tmax_lead = min(int(k+(window)),max(T))
        tmin_follow = k+tau
        tmax_follow = k+window+tau
        try : 
            X1 = differencing(np.array(x1[(x1['time']<=tmax_lead)&(x1['time']>=tmin_lead)]['r']), smooth=0.1)
            X2 = differencing(np.array(x2[(x2['time']<=tmax_follow)&(x2['time']>=tmin_follow)]['r']), smooth =0.1)
            distance, paths = dtw.warping_paths(X1, X2)
            DTW.append(distance)
        except Exception :
            DTW.append(np.nan)
        time.append(k)
        k=k + step
    return time,DTW

def create_dtw_by_time_df(tau_df,data,window,step,global_Tau=True):
    time_list,id_list,leader_list,DTW_list = [],[],[],[]
    if global_Tau==False:
        concatenated_df = pd.concat(tau_df)
        unique_ids_and_leaders = concatenated_df[['id', 'leader']].drop_duplicates()
        unique_ids_and_leaders
        for k in trange (len(unique_ids_and_leaders['id'])):
            leader = list(unique_ids_and_leaders['leader'])[k]
            id = list(unique_ids_and_leaders['id'])[k]
            tau_dic = prepareTaudf(data,concatenated_df,leader,id)
            time,DTW = test_car_following_by_time_variable_Tau(data,leader,id,tau_dic,window,step)
            id_list+= [id for k in range(len(time))]
            leader_list+= [leader for k in range(len(time))]
            time_list+=time
            DTW_list+=DTW
        df_out = pd.DataFrame({'id':id_list,'leader':leader_list,'time':time_list,'DTW':DTW_list})
        return df_out
    for k in trange (len(tau_df['id'])):
        leader = tau_df['leader'][k]
        id = tau_df['id'][k]
        tau = tau_df['Tau'][k]
        time,DTW = test_car_following_by_time(data,leader,id,tau,window,step)
        id_list+= [id for k in range(len(time))]
        leader_list+= [leader for k in range(len(time))]
        time_list+=time
        DTW_list+=DTW
    df_out = pd.DataFrame({'id':id_list,'leader':leader_list,'time':time_list,'DTW':DTW_list})
    return df_out

def define_CF(dtw_data,max_dtw):
    dtw_data['isCF'] = dtw_data['DTW'].apply(lambda x: True if x < max_dtw else False)
    return dtw_data

def determine_CF(trajsdf,DTW_df,max_dtw):
    DTW_df = define_CF(DTW_df,max_dtw)
    out = pd.DataFrame(columns = trajsdf.columns)
    out['CF'] = []
    for k in pd.unique(trajsdf['ID']):
        CF = []
        data = trajsdf[trajsdf['ID']==k]
        dtw_data = DTW_df[DTW_df['leader']==k]
        data.reset_index(inplace=True,drop=True)
        dtw_data.reset_index(inplace=True,drop=True)
        try:
            time_index = 0
            time = dtw_data['time'][time_index+1]
            for t in data['time']:
                if t<time:
                    CF.append(dtw_data['isCF'][time_index])
                elif t>time and time_index<len(dtw_data['time']):
                    if time_index + 1 < len(dtw_data['time']):
                        time_index+=1
                        if time_index + 1 < len(dtw_data['time']):
                            time = dtw_data['time'][time_index+1]
                    CF.append(dtw_data['isCF'][time_index])
                elif t>time and time_index==len(dtw_data['time']):
                    CF.append(dtw_data['isCF'][time_index])
        except Exception:
            pass
        if len(CF)<len(data['time']):
            CF+=[np.nan for k in range(len(data['time'])-len(CF))]
        data['CF']=CF
        out = pd.concat([out,data])
    return out