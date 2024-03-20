#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2023

This script analyzes the distribution of H_USER_ID.
It reads in all entropy files cached from the val+test split of GPT2-26M tabular model. 

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
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

def extract_entropy(dir, user_id, prov_type, calc_set='val_test', save=False):
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

    return df_entropy.loc[df_entropy['split_set']==calc_set, 'H_METRIC_NAME|REPORT_NAME']
    # return df_entropy['H_METRIC_NAME|REPORT_NAME']


## plotting function only supports when above function returns only a signle entropy column
def plot_histogram(df, job_label):
    # plot_histogram(df_H_USER_ID)
    bins = list(np.arange(0, df[0].max()+0.1, 0.1))
    # labels = ['2-5', '5-10', '10-20', '>20']
    df['H_count_bin'] = pd.cut(df[0], bins, labels=bins[:-1])
    for_plot = df.groupby('H_count_bin').agg(
        H_count_per_bin = pd.NamedAgg(0, aggfunc='count')
    ).reset_index()
    # display(for_plot)
    if job_label=='Attending':
        loc=0
        c='blue'
    else:
        loc=1
        c='red'
    bar_container = ax.plot(for_plot.H_count_bin, for_plot.H_count_per_bin, color=c, alpha=0.7, label=job_label)
    return for_plot
    
def plot_cleanup_save(ax, max_H, max_count, proj_dir, save=False):
    bins = list(np.arange(0, max_H+0.1, 0.1))
    ax.set(xticks=bins[:-1], xlim=[-0, max_H])
    # rounding ticks to multiples of 10000
    clean_number = round(max_count / 10000) * 10000
    ax.set(yticks=range(0,clean_number, round(clean_number/4/10000)*10000), ylim=[-5000,clean_number])
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Switch off ticks
    # ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    ax.legend(loc="upper right")
    # Set x-axis label
    # if i==1:
    ax.set_xlabel("Entropy for provider identifier values", labelpad=10, weight='bold', size=12)
    ax.set_ylabel("Count", labelpad=10, weight='bold', size=12)
    ax.tick_params(axis='x', rotation=0)
    # Draw horizontal axis lines
    vals = ax.get_yticks()
    for tick in vals:
        ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
    if save:
        fig.savefig(os.path.join(proj_dir, 'analysis/entropy_validation', 'plt_H_USER_ID_distribution.eps'))
        fig.savefig(os.path.join(proj_dir, 'analysis/entropy_validation', 'plt_H_USER_ID_distribution.png'))
    return

def draw_kdeplot(df, df2, proj_dir, save):
    ax = sns.distplot(df2[0], hist = False, kde_kws = {"shade" : True, "lw": 1}, color='red', label='APP')
    ax = sns.distplot(df[0], hist = False, kde_kws = {"shade" : True, "lw": 1}, color='blue', label='Attending')
    ax.set(xlim=[0, max(df[0].max(),df2[0].max())])
    ax.set_title("KDE Plot of Entropy (Color) and Normal Distribution (Black)")
    ax.set_xlabel("Entropy")
    ax.legend(fontsize=12)
    ax.set_title('')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    if save:
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/low_res/all_H_METRIC_REPORT_KDEplot.png"),bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/all_H_METRIC_REPORT_KDEplot.eps"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/all_H_METRIC_REPORT_KDEplot.tif"), bbox_inches='tight')
    ax.clear() 

    return



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
    parser.add_argument('--cache_data', help="Options:True, False | Default=False", default='False')
    parser.add_argument('--plot', help="Options:True, False | Default=False", default='False')
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

    if args.plot == 'False':
        include_plot = False
    elif args.plot == 'True':
        include_plot = True
    
    valid_provider_map=pd.read_csv(os.path.join(proj_dir, "../processed_data/Attending_and_APP_ICUshifts_mappable_job_map_combined.csv"))

    if include_plot:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        max_count = 0
        max_H = 0
    
    describe_H_USER_ID = pd.DataFrame()
    
    for job in ['Attending', 'APP']:
        print(f"++ Start processing {job}s... +++++++++++++++++++")
        pbar = tqdm(total=valid_provider_map.loc[valid_provider_map['PROV_TYPE']==job, 'USER_ID'].nunique(), desc="Users Processed", position=0, leave=True)
        prov_type = job
    
        
        df_H_USER_ID = pd.DataFrame()
        
        counter = 0
    
        # In SICU dataset directory, iterate through each subfolder named with unique USER_ID from our ICU provider cohort (65)
        for user_id in valid_provider_map.loc[valid_provider_map['PROV_TYPE']==job, 'USER_ID'].unique():
            # print(valid_provider_map.loc[valid_provider_map.USER_ID==user_id, :])
            # prov_type = valid_provider_map.loc[valid_provider_map.USER_ID==user_id, 'PROV_TYPE'].values[0]
        
            # # Access cached entropy from the out-of-sample set from the most parsimonious model (GPT2-26M)
            s_H_USER_ID = extract_entropy(data_dir, user_id, prov_type, calc_set='val_test', save=cache_data)
    
            df_H_USER_ID = pd.concat([df_H_USER_ID, s_H_USER_ID], axis=0)
            
            pbar.update(1)


        
        # draw_kdeplot(df_H_USER_ID, job, proj_dir, save=cache_data)
        
        describe_H = df_H_USER_ID.describe(percentiles=[.25, .5, .75, .95, .99])
        # describe_H.columns = [job+'_'+col_name for col_name in describe_H.columns]
        describe_H_USER_ID = pd.concat([describe_H_USER_ID, describe_H], axis=1)
        if include_plot:
            df_plot = plot_histogram(df_H_USER_ID, job_label=job)
            max_count = max(max_count, df_plot['H_count_per_bin'].max())
            max_H = max(max_H, df_H_USER_ID[0].max())
        if job=='Attending':
            df_attendings_H_USER_ID = df_H_USER_ID.copy()
    draw_kdeplot(df_attendings_H_USER_ID, df_H_USER_ID, proj_dir, save=cache_data)
    if include_plot:
        plot_cleanup_save(ax, max_H, max_count, proj_dir, save=cache_data)
    
    describe_H_USER_ID.columns = ['Attending', 'APP']
    # display(describe_H_USER_ID)
    
    print('\n',describe_H_USER_ID)
    print(f"\n95th percentile value of H_USER_ID:\n\t--Attending: {round(describe_H_USER_ID.loc['95%', 'Attending'],3)}\n\t--APP: {round(describe_H_USER_ID.loc['95%', 'APP'],3)}")
    print(f"99th percentile value of H_USER_ID:\n\t--Attending: {round(describe_H_USER_ID.loc['99%', 'Attending'],3)}\n\t--APP: {round(describe_H_USER_ID.loc['99%', 'APP'],3)}")
    
    
    
    
    if cache_data:
        print(f"\nSAVING H_USER_ID distribution...")
        describe_H_USER_ID.to_csv(os.path.join(proj_dir,'analysis/entropy_validation/tables', 'H_METRIC_REPORT_distribution.csv'))
        print("Done.")



