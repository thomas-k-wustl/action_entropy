#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2023

Case-Control matching for downstream statistical analyses

Control conditions of 3 non-routine transitions:
1. Patient switching transition
    Control Def: For each patient switch transition, we look at all other action transitions from the same provider, and extract all instances with the same action transition pair (ie, same antecedent and subsequent action metric identifiers) that doesn’t have an explicit switch between valid patient identifiers.
2. Non-inbox to inbox transition (ie, “to-inbox” transition; without patient switch)
    Control Def: For each “to-inbox” transition, we look at all other action transitions from the same provider, and extract all transition instances with the same antecedent action event but a subsequent non-inbox action (ie, non-inbox to non-inbox).
3. Inbox to non-inbox transition (ie, “from-inbox” transition; without patient switch)
    Control Def: For each “from-inbox” transition, we look at all other action transitions from the same provider, and extract all transition instances with the same antecedent action event but a subsequent inbox action (ie, inbox to inbox).

@author: Seunghwan (Nigel) Kim
@email: seunghwan.kim@wustl.edu

"""
import pandas as pd
import numpy as np
import argparse
from tqdm.auto import tqdm
import os
import datetime
import glob
import traceback
import warnings
warnings.filterwarnings("ignore")

def merge_entropy_to_accessLogs(user_path, df_entropy, calc_set='val_test', save=False):
    df_log = pd.read_csv(user_path + '/access_log_complete_ICU_shift.csv', parse_dates=['ACCESS_TIME'])
    # Sort by ACCESS_TIME and then by ACCESS_INSTANT
    df_log = df_log.sort_values(['ACCESS_TIME', 'ACCESS_INSTANT'])
    df_log = df_log.reset_index().drop('index', axis=1)
    ## Create a date column to use in the future as covariate for statistical analysis 
    # df_log['ACCESS_DATE'] = df_log['ACCESS_TIME'].dt.date
    ## Create a column to use in the future as covariate for statistical analysis: dailyabs # patients worked on EHR
    # df_log = df_log.merge(df_log.groupby('ACCESS_DATE')['PAT_ID'].nunique().reset_index().rename(columns={'PAT_ID':'n_uniq_PAT_ID'}), how='left', on='ACCESS_DATE')
    df_log = df_log.merge(df_log.groupby('workShift_date')['PAT_ID'].nunique().reset_index().rename(columns={'PAT_ID':'n_uniq_PAT_ID'}), how='left', on='workShift_date')

    try:
        df_log.loc[df_entropy.index, 'cached'] = 1
        df_log['cached'].fillna(0, inplace=True)
        df_log_entropy = pd.concat([df_log, df_entropy[df_entropy['split_set']==calc_set]], axis=1)
    except: # if there's off-by-one error for the row index of calculated Cross Entropy
        # return None
        print(traceback.format_exc())
    if save:
        print(f"--Saving complete audit logs with entropy joined for user {user_id}...")
        df_log_entropy.to_csv(os.path.join(user_path, 'access_log_complete_ICU_shift'+'_entropy2-gpt2-26_0M-'+prov_type+'.csv'))
    return df_log_entropy[df_log_entropy['cached']==1]

def preprocess_transition_pair_analysis(df):
    ## mark session start
    df.loc[df['H_METRIC_NAME|REPORT_NAME'].isnull(), 'session_start'] = 1
    ## create column for antecedant event
    df['metric_prev'] = df['METRIC_NAME'].shift(1)
    df['metric_report_prev'] = df['METRIC_NAME|REPORT_NAME'].shift(1)
    # ## create column for antecedent event's timestamp
    # df['ACCESS_TIME_prev'] = df['ACCESS_TIME'].shift(1)
    ## create column for timedelta of transition (seconds)
    df['timedelta_of_transition_s'] = (df['ACCESS_TIME'] - df['ACCESS_TIME'].shift(1)).dt.total_seconds()
    ## assign blank to antecedant event value at session start
    df.loc[df['session_start']==1, 'metric_prev'] = ''
    df.loc[df['session_start']==1, 'metric_report_prev'] = ''
    ## assign blank to transition timedelta value at session start
    df.loc[df['session_start']==1, 'timedelta_of_transition'] = None
    ## assign blank to missing PAT_IDs at session start
    df.loc[(df['session_start']==1) &\
              (df['PAT_ID'].isnull()),
              'PAT_ID'] = ''
    ## imputing missing PAT_ID values
    ## -- Strategy 1: forward fill missing PAT_ID (valid values from the immediate past propagates into future)
    # df['PAT_ID'].fillna(method='ffill', inplace=True)
    ## -- Strategy 2: no imputation
    df['PAT_ID'].fillna('', inplace=True)
    ## create column for antecedant PAT_ID
    df['pat_prev'] = df['PAT_ID'].shift(1).fillna('')
    ## assign blank to antecedant PAT_ID value at session start
    df.loc[df['session_start']==1, 'pat_prev'] = ''
    return df

def catch_pat_switch(df):
    df.loc[(df['PAT_ID']!='') & (df['pat_prev']!='') &\
            (df['PAT_ID']!=df['pat_prev']),
        'pat_switch'] = 1
    # n_pat_switch = len(df[df['pat_switch']==1])
    return df#, n_pat_switch

def match_pat_switch_controls(df):
    uniq_pat_switch_event_pairs = df.loc[df['pat_switch']==1, ['METRIC_NAME|REPORT_NAME', 'metric_report_prev']].drop_duplicates().set_index(['METRIC_NAME|REPORT_NAME', 'metric_report_prev']).index
    # display(len(uniq_pat_switch_event_pairs.values))
    df.loc[(df['pat_switch']!=1) &\
            (df.set_index(['METRIC_NAME|REPORT_NAME', 'metric_report_prev']).index.isin(uniq_pat_switch_event_pairs)),
        'pat_switch_control'] = 1
    return df

def catch_inBasket_transition(df, to_inbasket=True):
    global inBasket_metrics_dict
    if to_inbasket: 
        df.loc[(df['METRIC_NAME'].isin(inBasket_metrics_dict.loc[inBasket_metrics_dict['to_IB']==1, 'METRIC_NAME'])) &\
                (~df['metric_prev'].isin(inBasket_metrics_dict['METRIC_NAME'])) &\
                (df['pat_switch']!=1) &\
                (df['metric_prev']!=''), #subsequent action is in basket, antecedent action is non-IB. NON-PATIENT switching transitions only!
            'to_inbasket'] = 1
        # display(df.loc[df['to_inbasket']==1, ['ACCESS_TIME', 'session_start', 'METRIC_NAME|REPORT_NAME', 'metric_prev', 'to_inbasket']])
        # display(df.loc[df['to_inbasket']==1, 'metric_prev'].value_counts())
    else:
        df.loc[(df['metric_prev'].isin(inBasket_metrics_dict.loc[inBasket_metrics_dict['from_IB']==1, 'METRIC_NAME'])) &\
                (~df['METRIC_NAME'].isin(inBasket_metrics_dict['METRIC_NAME'])) &\
                (df['pat_switch']!=1) &\
                (df['METRIC_NAME']!=''), #antecedent action is in basket, subsequent action is non-IB. NON-PATIENT switching transitions only!
            'from_inbasket'] = 1
        # display(df.loc[df['from_inbasket']==1, ['ACCESS_TIME', 'session_start', 'METRIC_NAME|REPORT_NAME', 'metric_prev', 'from_inbasket']])
    return df#, n_pat_switch

def match_inBasket_transition_controls(df, to_inbasket=True):
    global inBasket_metrics_dict
    if to_inbasket:
        df.loc[(df['metric_report_prev'].isin(df.loc[df['to_inbasket']==1, 'metric_report_prev'])) &\
                (~df['METRIC_NAME|REPORT_NAME'].isin(df.loc[df['to_inbasket']==1, 'METRIC_NAME|REPORT_NAME'])) &\
                (df['pat_switch']!=1), #same antecedent(non-IB), non-IB subsequent
            'to_inbasket_control'] = 1
        # display(df.loc[df['to_inbasket_control']==1, ['ACCESS_TIME', 'session_start', 'METRIC_NAME|REPORT_NAME', 'metric_prev', 'to_inbasket_control']])
    else:
        df.loc[(df['metric_report_prev'].isin(df.loc[df['from_inbasket']==1, 'metric_report_prev'])) &\
                (df['METRIC_NAME'].isin(inBasket_metrics_dict['METRIC_NAME'])) &\
                (df['pat_switch']!=1), #same antecedent(IB), IB subsequent
            'from_inbasket_control'] = 1
        # display(df.loc[df['from_inbasket_control']==1, ['ACCESS_TIME', 'session_start', 'METRIC_NAME|REPORT_NAME', 'metric_prev', 'from_inbasket_control']])
    return df

def build_cohort(dir, user_id, prov_type, save=False):
    ''' input:  dir/user_id: directory containing the access_log_complete.csv and file that has the cached cross-entroppy values
    '''
    user_path = os.path.join(dir, user_id)
    try:
        df_entropy = pd.read_csv(user_path + '/entropy2-gpt2-26_0M-'+prov_type+'.csv', names=['H_METRIC_NAME|REPORT_NAME', 'H_PAT_ID', 'H_ACCESS_TIME', 'H_USER_ID', 'split_set'], header=0, index_col=0)
        print(f"Loading cached entropy file for user {user_id}...")
    except:
        if len(glob.glob(user_path+'/entropy2*'+prov_type+'.csv')) == 0:
            print(f"No cached entropy files for user {user_id}")
        print(traceback.format_exc())
        return None

    # merge cached event-wise entropies to ICU audit logs & return only the cached sessions
    df_log_cached_entropy = merge_entropy_to_accessLogs(user_path, df_entropy, calc_set='val_test', save=save)

    # modify dataframe to make bigram-based filtering easier
    df_log_cached_entropy = preprocess_transition_pair_analysis(df_log_cached_entropy)
       
    # catch patient switching transition event CASES
    df_log_cached_entropy = catch_pat_switch(df_log_cached_entropy)
    # match with CONTROLS
    df_log_cached_entropy = match_pat_switch_controls(df_log_cached_entropy)
    # keep full access logs for valid CASES & CONTROLS
    ## 1. Patient switching
    df_log_pat_switch_case_control = df_log_cached_entropy[(df_log_cached_entropy['pat_switch'].notnull()) |\
                                                           (df_log_cached_entropy['pat_switch_control'].notnull())]
    if len(df_log_pat_switch_case_control[df_log_pat_switch_case_control['pat_switch_control']==1]) > 0:
        df_log_pat_switch_case_control.loc[df_log_pat_switch_case_control['pat_switch']==1, 'Case'] = 1
        df_log_pat_switch_case_control.loc[df_log_pat_switch_case_control['pat_switch_control']==1, 'Case'] = 0

    # display(df_log_cached_entropy.loc[df_log_cached_entropy['pat_switch']==1, 
    #         ['ACCESS_TIME', 'session_start', 'METRIC_NAME|REPORT_NAME', 'event_prev', 'PAT_ID', 'pat_prev',
    #          'pat_switch', 'pat_switch_control']])
    # display(df_log_cached_entropy.loc[df_log_cached_entropy['pat_switch_control']==1, 
    #         ['ACCESS_TIME', 'session_start', 'METRIC_NAME|REPORT_NAME', 'event_prev', 'PAT_ID', 'pat_prev',
    #          'pat_switch', 'pat_switch_control']])
    # display(df_log_pat_switch_case_control)

    # catch in basket transition event CASES
    df_log_cached_entropy = catch_inBasket_transition(df_log_cached_entropy, to_inbasket=True)
    df_log_cached_entropy = catch_inBasket_transition(df_log_cached_entropy, to_inbasket=False)
    # match with CONTROLS
    df_log_cached_entropy = match_inBasket_transition_controls(df_log_cached_entropy, to_inbasket=True)
    df_log_cached_entropy = match_inBasket_transition_controls(df_log_cached_entropy, to_inbasket=False)
    # keep full access logs for valid CASES & CONTROLS
    ## 2. To- In Basket
    df_log_to_IB_case_control = df_log_cached_entropy[(df_log_cached_entropy['to_inbasket'].notnull()) |\
                                                        (df_log_cached_entropy['to_inbasket_control'].notnull())]
    if len(df_log_to_IB_case_control[df_log_to_IB_case_control['to_inbasket_control']==1]) > 0:
        df_log_to_IB_case_control.loc[df_log_to_IB_case_control['to_inbasket']==1, 'Case'] = 1
        df_log_to_IB_case_control.loc[df_log_to_IB_case_control['to_inbasket_control']==1, 'Case'] = 0
    ## 3. From- In Basket
    df_log_from_IB_case_control = df_log_cached_entropy[(df_log_cached_entropy['from_inbasket'].notnull()) |\
                                                        (df_log_cached_entropy['from_inbasket_control'].notnull())]
    if len(df_log_from_IB_case_control[df_log_from_IB_case_control['from_inbasket_control']==1]) > 0:
        df_log_from_IB_case_control.loc[df_log_from_IB_case_control['from_inbasket']==1, 'Case'] = 1
        df_log_from_IB_case_control.loc[df_log_from_IB_case_control['from_inbasket_control']==1, 'Case'] = 0

    
    print("--Case-Control matching finished.")

    
    return df_log_pat_switch_case_control, df_log_to_IB_case_control, df_log_from_IB_case_control





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
    parser.add_argument('--cache_data', help="Options:True, False | Default=True", default=False)
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

    if args.cache_data == 'False':
        cache_data = False
    elif args.cache_data == 'True':
        cache_data = True

    valid_provider_map=pd.read_csv(os.path.join(proj_dir, "../processed_data/Attending_and_APP_ICUshifts_mappable_job_map_combined.csv"))
    inBasket_metrics_dict=pd.read_excel(os.path.join(data_dir, "../inBasket_metrics.xlsx"))
    
    

    

    for job in ['Attending', 'APP']:
        print(f"++ Start processing {job}s... +++++++++++++++++++")
        pbar = tqdm(total=valid_provider_map.loc[valid_provider_map['PROV_TYPE']==job, 'USER_ID'].nunique(), desc="Users Processed", position=0, leave=True)
        df_all_pat_switch_case_control = pd.DataFrame()
        df_all_to_IB_case_control = pd.DataFrame()
        df_all_from_IB_case_control = pd.DataFrame()
    
        pat_switch_no_match = []
        toIB_no_match = []
        fromIB_no_match = []
        # In SICU dataset directory, iterate through each subfolder named with unique USER_ID from our ICU provider cohort (65)
        for user_id in valid_provider_map.loc[valid_provider_map['PROV_TYPE']==job, 'USER_ID'].unique():
            # print(valid_provider_map.loc[valid_provider_map.USER_ID==user_id, :])
            prov_type = valid_provider_map.loc[valid_provider_map.USER_ID==user_id, 'PROV_TYPE'].values[0]
        
            # # Access cached entropy from the out-of-sample set from the most parsimonious model (GPT2-26M)
            pat_switch_case_control, to_IB_case_control, from_IB_case_control = build_cohort(data_dir, user_id, prov_type, save=cache_data)
    
            # If at least 1 control sample identified, add it to the collection.
            if len(pat_switch_case_control[pat_switch_case_control['pat_switch_control']==1]) > 0:
                df_all_pat_switch_case_control = pd.concat([df_all_pat_switch_case_control, pat_switch_case_control], axis=0)
            else:
                pat_switch_no_match.append(user_id)
            if len(to_IB_case_control[to_IB_case_control['to_inbasket_control']==1]) > 0:
                df_all_to_IB_case_control = pd.concat([df_all_to_IB_case_control, to_IB_case_control], axis=0)
            else:
                toIB_no_match.append(user_id)
            if len(from_IB_case_control[from_IB_case_control['from_inbasket_control']==1]) > 0:
                df_all_from_IB_case_control = pd.concat([df_all_from_IB_case_control, from_IB_case_control], axis=0)
            else:
                fromIB_no_match.append(user_id)
            pbar.update(1)
            # break
        
        # display(df_all_pat_switch_case_control, df_all_to_IB_case_control, df_all_from_IB_case_control)
        
        # make sure to remove any row without USER_ID value (idk why these even happen in the original audit logs)
        df_all_pat_switch_case_control = df_all_pat_switch_case_control[df_all_pat_switch_case_control['USER_ID'].notnull()]
        df_all_to_IB_case_control = df_all_to_IB_case_control[df_all_to_IB_case_control['USER_ID'].notnull()]
        df_all_from_IB_case_control = df_all_from_IB_case_control[df_all_from_IB_case_control['USER_ID'].notnull()]
        
        if cache_data:
            print(f"SAVING ALL CASE-CONTROL DATASETS...")
            # if job=='APP':
            #     print(df_all_to_IB_case_control.USER_ID.unique())
            # # make sure no duplicate index
            # df_all_pat_switch_case_control = df_all_pat_switch_case_control.reset_index()
            # df_all_to_IB_case_control = df_all_to_IB_case_control.reset_index()
            # df_all_from_IB_case_control = df_all_from_IB_case_control.reset_index()
            # # make sure all USER_IDs are strings...
            # df_all_pat_switch_case_control['USER_ID'] = df_all_pat_switch_case_control['USER_ID'].astype("string")
            # df_all_to_IB_case_control['USER_ID'] = df_all_pat_switch_case_control['USER_ID'].astype("string")
            # df_all_from_IB_case_control['USER_ID'] = df_all_pat_switch_case_control['USER_ID'].astype("string")
            # if job=='APP':
            #     print(df_all_to_IB_case_control.USER_ID.unique())
        
            df_all_pat_switch_case_control.to_csv(os.path.join(proj_dir, "processed_data/"+job+"_pat_switch_cases_controls.csv"))
            df_all_to_IB_case_control.to_csv(os.path.join(proj_dir, "processed_data/"+job+"_to_IB_cases_controls.csv"))
            df_all_from_IB_case_control.to_csv(os.path.join(proj_dir, "processed_data/"+job+"_from_IB_cases_controls.csv"))
            print("Done.")
        
        print(f"1. Total # patient switch transition CASES: {len(df_all_pat_switch_case_control[df_all_pat_switch_case_control['pat_switch']==1])}")
        print(f"\tTotal # patient switch transition CONTROLS: {len(df_all_pat_switch_case_control[df_all_pat_switch_case_control['pat_switch_control']==1])}")
        print(f"\tTotal # USERs unable to match: {len(pat_switch_no_match)} ({pat_switch_no_match})")
        print(f"2. Total # to-IB transition CASES: {len(df_all_to_IB_case_control[df_all_to_IB_case_control['to_inbasket']==1])}")
        print(f"\tTotal # to-IB transition CONTROLS: {len(df_all_to_IB_case_control[df_all_to_IB_case_control['to_inbasket_control']==1])}")
        print(f"\tTotal # USERs unable to match: {len(toIB_no_match)} ({toIB_no_match})")
        print(f"3. Total # from-IB transition CASES: {len(df_all_from_IB_case_control[df_all_from_IB_case_control['from_inbasket']==1])}")
        print(f"\tTotal # from-IB transition CONTROLS: {len(df_all_from_IB_case_control[df_all_from_IB_case_control['from_inbasket_control']==1])}")
        print(f"\tTotal # USERs unable to match: {len(fromIB_no_match)} ({fromIB_no_match})")

