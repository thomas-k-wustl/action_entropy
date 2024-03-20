#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created  on Jan 25 2024

Take the preprocess audit logs (output from preproccess_audit_logs.py) and calculate additional descriptive statistics.

@author: Seunghwan (Nigel) Kim
@email: seunghwan.kim@wustl.edu

"""
import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm
import datetime
import os
import sys
import warnings
warnings.filterwarnings("ignore")
    
def calc_sessions(df):
    # reset time deltas
    # 5min cap gap session interval
    SESSION_INTERVAL = 5 * 60 
    time_deltas = df.loc[:, 'ACCESS_INSTANT'].diff(periods=-1)*-1
    df['time_delta'] = np.where((time_deltas > SESSION_INTERVAL), np.nan, time_deltas)
    num_sesh = df['time_delta'].isnull().sum()
    return num_sesh

def calc_EHR_use_time(df):
    EHR_duration_sec = 0
    for workShift_date in df['workShift_date'].unique():
        # first calculaate daily EHR use time
        df_shift_logs = df.loc[df['workShift_date']==workShift_date, :].sort_values(['ACCESS_TIME', 'ACCESS_INSTANT'])
        # add them up cumulatively
        EHR_duration_sec += (df_shift_logs.iloc[-1, 1] - df_shift_logs.iloc[0, 1]).seconds
    # convert seconds to hours
    total_EHR_use_duration_hour = EHR_duration_sec/3600
    return total_EHR_use_duration_hour

def describe_audit_logs(user_id, user_path):
    ''' 
    Note
            AM shift: grab logs between 12am-12am
            PM shift: grab logs between 12pm-12pm
    '''
    df_user = pd.read_csv(user_path + '/access_log_complete_ICU_shift.csv')
    df_user['ACCESS_TIME'] = pd.to_datetime(df_user['ACCESS_TIME'])

    # calculate number of events in the log
    num_logs = len(df_user)
    # calc number of shifts shifts in the log
    num_shifts = df_user['workShift_date'].nunique()
    # calculate number of sessions
    num_sessions = calc_sessions(df_user)
    # calculate daily EHR use duration... and then sum to total
    total_EHR_use_duration_hour = calc_EHR_use_time(df_user)
    # extract unique patient identifiers
    uniq_pat_list = df_user.loc[df_user['PAT_ID'].notnull(), 'PAT_ID'].unique().tolist()
    # calculate number of unique pats per shift & sum it up for all shifts
    # total_num_pat_seen = np.sum(df_user[df_user['PAT_ID'].notnull()].groupby('workShift_date')['PAT_ID'].nunique().values)
    num_pat_seen_list = df_user[df_user['PAT_ID'].notnull()].groupby('workShift_date')['PAT_ID'].nunique().values.tolist()
    # print(total_num_pat_seen)
    # print(type(total_num_pat_seen))

    
    return num_logs, num_shifts, num_sessions, total_EHR_use_duration_hour, uniq_pat_list, num_pat_seen_list
        

if __name__ == '__main__':
    '''
    Run the code in shell:
    python3 FILENAME.py 
    OR
    python3 FILENAME.py --datapath FILEPATH --projpath FILEPATH
    OR
    python3 FILENAME.py --os OS
    '''
    parser = argparse.ArgumentParser(description='Explore available data.')
    parser.add_argument('--datapath', help='OPTIONAL | Default=none | directory containing subfolders with audit log data', default='none')
    parser.add_argument('--projpath', help='OPTIONAL | Default=none | project directory', default='none')
    parser.add_argument('--os', help="Options:windows, mac, argonaute | Default=mac | Your current machine's operating system (windows/mac). In case when using remote compute server, specify argonaute).", default='mac')
    args = parser.parse_args()

    # Specify where the audit log data is
    if args.datapath == "none":
        if args.os == "windows": # Windows OS
            data_dir = os.path.join("Z:", "Active", "icu_ehr_logs", "raw_data", "2019")
        elif args.os == "mac":
            data_dir = os.path.join("/Volumes/Active/icu_ehr_logs", "raw_data/2019")
        elif args.os == "argonaute":
            data_dir = os.path.join("/home/skim/ris_share", "Active/icu_ehr_logs", "raw_data/2019")
        else:
            sys.exit("You MUST choose from the available options.")
    else:
        data_dir = args.datapath
        
    # Specify the project directory
    if args.projpath == "none":
        if args.os == "windows": # Windows OS
            proj_dir = os.path.join("Z:", "Active", "icu_ehr_logs/entropy_project")
        elif args.os == "mac": # Mac OS
            proj_dir = "/Volumes/Active/icu_ehr_logs/entropy_project"
        elif args.os == "argonaute":
            proj_dir = os.path.join("/home/skim/ris_share", "Active/icu_ehr_logs/entropy_project")
        else:
            sys.exit("You MUST choose from the available options.")
    else:
        proj_dir = args.projpath

    today_date = str(datetime.date.today())
    
    valid_provider_map=pd.read_csv("Z:/Active/icu_ehr_logs/processed_data/providers_ICUshifts_mappable.csv")
    valid_provider_map['Assign Date'] = pd.to_datetime(valid_provider_map['Assign Date'])
    
    df_table = pd.DataFrame(columns=['attending', 'APP', 'All'])
    df_table.loc['num_logs', :] = [0,0,0]
    df_table.loc['num_shifts', :] = [0,0,0]
    df_table.loc['num_sessions', :] = [0,0,0]
    df_table.loc['EHR_use_hours', :] = [0,0,0]
    # df_table.loc['num_uniq_patients', :] = [0,0,0]
    # df_table.loc['total_num_uniq_pat_seen_perShift', :] = [0,0,0]

    dict_uniq_patID = {'attending': set(), 'APP': set(), 'All': set()}
    dict_total_num_pat_seen = {'attending': [], 'APP': [], 'All': []}
    
    pbar = tqdm(total=valid_provider_map['USER_ID'].nunique(), desc="Users Processed", position=0, leave=True)
    # In SICU dataset directory, iterate through each subfolder named with unique USER_ID
    for user_id in valid_provider_map['USER_ID'].unique():
        # print(user_id)
        user_map = valid_provider_map[valid_provider_map['USER_ID']==user_id]
        num_logs, num_shifts, num_sessions, EHR_use_hours, uniq_pat_list, num_pat_seen_list = describe_audit_logs(user_id, 
                                                                                             os.path.join(data_dir, user_id)
                                                                                            )
        
        if user_map['PROV_TYPE'].values[0] in ['Physician', 'Anesthesiologist']:
            job_type = 'attending'
        elif user_map['PROV_TYPE'].values[0] in ['Nurse Practitioner', 'Physician Assistant']:
            job_type = 'APP'

        # add patient list to unique patient set for the specified job_type
        dict_uniq_patID[job_type].update(uniq_pat_list)
        dict_total_num_pat_seen[job_type] += num_pat_seen_list
        
        df_table.loc[:, job_type] += [num_logs, num_shifts, num_sessions, EHR_use_hours]
        df_table.loc[:, 'All'] += [num_logs, num_shifts, num_sessions, EHR_use_hours]
        
        pbar.update(1)
        # break

    # calculating total count of unique patients seen by attendings, APPs and combined (unique set)
    df_table.loc['num_uniq_patients', 'attending'] = len(dict_uniq_patID['attending'])
    df_table.loc['num_uniq_patients', 'APP'] = len(dict_uniq_patID['APP'])
    
    dict_uniq_patID['All'].update(dict_uniq_patID['attending'])
    dict_uniq_patID['All'].update(dict_uniq_patID['APP'])
    df_table.loc['num_uniq_patients', 'All'] = len(dict_uniq_patID['All'])

    # calculating mean +/- std deviation of total count of unique patients seen per shift
    ## pad lists with 0 (add in zeros for valid work shifts with no actions without valid PAT_ID attached)
    dict_total_num_pat_seen['attending'] += ([0] * int(df_table.loc['num_shifts', 'attending']-len(dict_total_num_pat_seen['attending'])))
    dict_total_num_pat_seen['APP'] += ([0] * int(df_table.loc['num_shifts', 'APP']-len(dict_total_num_pat_seen['APP'])))
    ## begin calculation
    dict_total_num_pat_seen['All'] += dict_total_num_pat_seen['attending']
    dict_total_num_pat_seen['All'] += dict_total_num_pat_seen['APP']
    df_table.loc['total_num_uniq_pat_seen_perShift', :] = [np.sum(dict_total_num_pat_seen['attending']),
                                                           np.sum(dict_total_num_pat_seen['APP']),
                                                           np.sum(dict_total_num_pat_seen['All'])]
    df_table.loc['avg_num_uniq_pat_perShift', :] = [np.mean(dict_total_num_pat_seen['attending']),
                                                    np.mean(dict_total_num_pat_seen['APP']),
                                                    np.mean(dict_total_num_pat_seen['All'])]
    df_table.loc['std_num_uniq_pat_perShift', :] = [np.std(dict_total_num_pat_seen['attending']),
                                                    np.std(dict_total_num_pat_seen['APP']),
                                                    np.std(dict_total_num_pat_seen['All'])]
    [print(len(x)) for x in dict_total_num_pat_seen.values()]
    # save
    df_table.to_csv(os.path.join(proj_dir, "processed_data/descriptive_statistics_"+today_date+".csv"), index=True)
    
    
    print(f"total available # audit log events: {df_table.loc['num_logs', 'All']}")
    print(df_table)

    

