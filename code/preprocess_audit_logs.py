#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 26 2023

@author: Seunghwan (Nigel) Kim
@email: seunghwan.kim@wustl.edu

"""
import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm
import os
import sys
import warnings
warnings.filterwarnings("ignore")

def investigate_logs(df):
    df['ACCESS_TIME_t'] = df['ACCESS_TIME'].dt.time.astype(str).str[:5]
    inbasket_distr = df.loc[df['METRIC_NAME|REPORT_NAME']=='In Basket message created', 'ACCESS_TIME_t'].value_counts().reset_index()
    generic_order_distr = df.loc[df['METRIC_NAME|REPORT_NAME']=='Order transmitted | GENERIC ORDER COMPLETION REPORT', 'ACCESS_TIME_t'].value_counts().reset_index()
    dummy_distr = df.loc[df['METRIC_NAME|REPORT_NAME']=='Order transmitted | DUMMY REPORT', 'ACCESS_TIME_t'].value_counts().reset_index()
    
    return [inbasket_distr, generic_order_distr, dummy_distr]
    

def preprocess_audit_logs(user_id, user_shift_map, user_path, save, autotrigger_investigate):
    ''' 
    Note
            AM shift: grab logs between 12am-12am
            PM shift: grab logs between 12pm-12pm
    '''
    df_user = pd.read_csv(user_path + '/access_log_complete.csv')
    df_user['ACCESS_TIME'] = pd.to_datetime(df_user['ACCESS_TIME'])

    user_ICU_logs = pd.DataFrame()
    for _,row in user_shift_map.iterrows():
        shift_time_str = row['shift_time']
        # select the time window of interest btwn timestamp and timestamp - window -- Modified to reflect AM vs PM shifts
        if row['shift_time']=='AM':
            ICU_logs = df_user[(df_user.ACCESS_TIME < row['Assign Date'] + pd.DateOffset(days=1)) & (df_user.ACCESS_TIME >= row['Assign Date'])]
        elif row['shift_time']=='PM':
            ICU_logs = df_user[(df_user.ACCESS_TIME >= row['Assign Date'] + pd.DateOffset(hours=12)) & (df_user.ACCESS_TIME < row['Assign Date'] + pd.DateOffset(hours=36))]
        elif row['shift_time']=='Unknown':
            # Sunny: these Swing/Rounding NPs are usually just helping out, and aren't assigned to a specific AM/PM shift, but they are most likely AM shift.
            # Mapping them to AM shifts.
            ICU_logs = df_user[(df_user.ACCESS_TIME < row['Assign Date'] + pd.DateOffset(days=1)) & (df_user.ACCESS_TIME >= row['Assign Date'])]
            shift_time_str = 'AM'
        ICU_logs['workShift_date'] = str(row['Assign Date'])+'_'+shift_time_str
        user_ICU_logs = pd.concat([user_ICU_logs, ICU_logs], axis=0)
    
    user_ICU_logs['REPORT_NAME'] = ' | '+user_ICU_logs['REPORT_NAME']
    user_ICU_logs['REPORT_NAME'].fillna("", inplace=True)
    user_ICU_logs['METRIC_NAME|REPORT_NAME'] = user_ICU_logs['METRIC_NAME']+user_ICU_logs['REPORT_NAME']
    
    if save==True:
        user_ICU_logs.to_csv(os.path.join(user_path, 'access_log_complete_ICU_shift.csv'), index=False)
    elif save=='shift':
        user_ICU_logs.to_csv(os.path.join(user_path, 'access_log_complete_ICU_shift_dtl.csv'), index=False)
    
    num_logs = len(user_ICU_logs)
    
    if autotrigger_investigate:
        df_distr = investigate_logs(user_ICU_logs)
        return num_logs, df_distr
    
    return num_logs, _
        

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
    parser.add_argument('--cache_data', help="Options:True, False, Shift | Default=False", default=False)
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

    if args.cache_data == 'True':
        cache_data = True
    elif args.cache_data == 'shift':
        cache_data = 'shift'
    else:
        cache_data = False

    valid_provider_map=pd.read_csv(os.path.join(proj_dir, "../processed_data/providers_ICUshifts_mappable.csv"))
    valid_provider_map['Assign Date'] = pd.to_datetime(valid_provider_map['Assign Date'])
    
    
    investigate=False
    if investigate:
        df_IB = pd.DataFrame()
        df_gen_order = pd.DataFrame()
        df_dummy = pd.DataFrame()
    
    tot_num_logs = 0
    pbar = tqdm(total=valid_provider_map['USER_ID'].nunique(), desc="Users Processed", position=0, leave=True)
    # In SICU dataset directory, iterate through each subfolder named with unique USER_ID
    for user_id in valid_provider_map['USER_ID'].unique():
        # print(user_id)
        user_map = valid_provider_map[valid_provider_map['USER_ID']==user_id]
        num_logs, df_distr = preprocess_audit_logs(user_id, 
                                                   user_map, 
                                                   os.path.join(data_dir, user_id),
                                                   save=cache_data,
                                                   autotrigger_investigate=investigate
                                                  )
        
        
        tot_num_logs += num_logs
        
        if investigate:
            df_IB = pd.concat([df_IB, df_distr[0]], axis=0)
            df_gen_order = pd.concat([df_gen_order, df_distr[1]], axis=0)
            df_dummy = pd.concat([df_dummy, df_distr[2]], axis=0)
        
        pbar.update(1)
        
    
    print(f"total available # audit log events: {tot_num_logs}")
    if investigate:
        df_IB = df_IB.groupby('ACCESS_TIME_t').sum().reset_index()
        df_gen_order = df_gen_order.groupby('ACCESS_TIME_t').sum().reset_index()
        df_dummy = df_dummy.groupby('ACCESS_TIME_t').sum().reset_index()
        print(df_IB.sort_values('count', ascending=False))
        print(df_gen_order.sort_values('count', ascending=False))
        print(df_dummy.sort_values('count', ascending=False))
        df_IB.to_csv(os.path.join(proj_dir, "../processed_data/inbasket_distribution.csv"), index=False)
        df_gen_order.to_csv(os.path.join(proj_dir, "../processed_data/generic_order_report_distribution.csv"), index=False)
        df_dummy.to_csv(os.path.join(proj_dir, "../processed_data/dummy_report_distribution.csv"), index=False)


