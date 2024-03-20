#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 2024

Comparison test (between-subjects design)
(eg. case-control come from independent samples)

**Question**: "Is non-routine transition associated with higher entropy?"

**H_0**: true difference between these group medians is zero

Mann-Whitney U tests across all matched case-control groups (2 clinician groups, 3 case-control criteria)

(Recap)
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
from scipy.stats import shapiro, ranksums, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

def check_normality(df):
    print("Checking normality of data... (Shapiro-Wilk test; p>0.05 indicates normal distribution)")
    test_cases = shapiro(df.loc[df['Case']==1, 'H_METRIC_NAME|REPORT_NAME'])
    test_controls = shapiro(df.loc[df['Case']==0, 'H_METRIC_NAME|REPORT_NAME'])
    print(f"\tCases: {test_cases}")
    print(f"\tControls: {test_controls}")
    if (test_cases[1] > 0.05) & (test_controls[1] > 0.05):
        print("Data normally distributed. Proceed with Independent t-test (parametric).")
        return True
    else:
        print("Data not normally distributed. Proceed with Wilcoxon Rank-Sum test (aka Mann-Whitney U test; non-parametric).")
        return False

def draw_kdeplot(df, job, interruption, proj_dir, save):
    # ax = sns.distplot(df.loc[df['Case']==1, 'H_METRIC_NAME|REPORT_NAME'], hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm, color='blue', label='Case')
    # ax = sns.distplot(df.loc[df['Case']==0, 'H_METRIC_NAME|REPORT_NAME'], hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm, color='red', label='Control')
    ax = sns.distplot(df.loc[df['Case']==1, 'H_METRIC_NAME|REPORT_NAME'], hist = False, kde_kws = {"shade" : True, "lw": 1}, color='blue', label='Case')
    ax = sns.distplot(df.loc[df['Case']==0, 'H_METRIC_NAME|REPORT_NAME'], hist = False, kde_kws = {"shade" : True, "lw": 1}, color='red', label='Control')
    ax.set(xlim=[0, max(df.loc[df['Case']==1, 'H_METRIC_NAME|REPORT_NAME'].max(),df.loc[df['Case']==0, 'H_METRIC_NAME|REPORT_NAME'].max())])
    ax.set_title("KDE Plot of Entropy (Color) and Normal Distribution (Black)")
    ax.set_xlabel("Entropy")
    ax.legend(fontsize=12)
    ax.get_legend().set_title(job, prop = { "size": 12 })
    ax.set_title('')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    if save:
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/low_res/"+job+"_"+interruption.split('_cases')[0]+"_KDEplot.png"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+job+"_"+interruption.split('_cases')[0]+"_KDEplot.eps"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+job+"_"+interruption.split('_cases')[0]+"_KDEplot.tif"), bbox_inches='tight')
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


    for job in ['Attending', 'APP']:
        print(f"+++++++++++++++++++ Processing {job}s... +++++++++++++++++++")
        for interruption in ['pat_switch_cases_controls', 'to_IB_cases_controls', 'from_IB_cases_controls']:
            print(f"\n++Testing {interruption}...")
            df = pd.read_csv(os.path.join(proj_dir, "processed_data/"+job+'_'+interruption+".csv"))
            normal = check_normality(df)
            if not normal:
                # Perform Wilcoxon Signed-rank test as a non-parametric comparison test                
                ## get basic descriptions of data distribution in the case and control groups
                draw_kdeplot(df, job, interruption, proj_dir, save=cache_data)
                case_control_distribution = pd.concat([df.loc[df['Case']==1, 'H_METRIC_NAME|REPORT_NAME'].describe(), df.loc[df['Case']==0, 'H_METRIC_NAME|REPORT_NAME'].describe()], axis=1)
                case_control_distribution.columns = ['Cases', 'Controls']
                display(case_control_distribution)
    
                ## perform comparison test
                ### two-sided Mann-Whitney U test (H_a: the distributions are not equal)
                res = mannwhitneyu(x = df.loc[df['Case']==1, 'H_METRIC_NAME|REPORT_NAME'], 
                                   y = df.loc[df['Case']==0, 'H_METRIC_NAME|REPORT_NAME'],
                                   alternative = 'two-sided')

                print(f"Two-sided MWU test: {res}")
                if res[1]<0.05:
                    print("\t*Reject H_0. Two groups are statistically significantly different")
                
                ### one-sided Mann-Whitney U test (H_a: median entropy of non-routine transition is greater than that of routine transitions)
                res = mannwhitneyu(x = df.loc[df['Case']==1, 'H_METRIC_NAME|REPORT_NAME'], 
                                   y = df.loc[df['Case']==0, 'H_METRIC_NAME|REPORT_NAME'],
                                   alternative = 'greater')

                print(f"One-sided MWU test: {res}")
                if res[1]<0.05:
                    print(f"\t*Reject H_0. Median entropy of {interruption.split('_cases')[0]} transitions is statisticallly significantly greater than that of routine transitions")
        #     break
        # break
    

    