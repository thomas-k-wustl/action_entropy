#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 2023

# Multivariable linear regression

**Question**: "Is non-routine transition (ie, cases) associated with higher entropy?"

**Independent variable**
Non-routine transition (binary variable; eg, patient switch)

**Outcome variable**
Entropy of transition (continuous)

**Additional covariates**

*Action-transition level characteristics*
1. Timedelta between transition
   
*User-level or Day-level characteristics*

2. USER_ID (clustering variable)
3. DATE (clustering variable)
4. daily workload
   * counts of unique patients worked on EHR

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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def draw_boxplot(df, scaled_entropy, job, interruption, proj_dir, save):
    global today_date
    boxplot = df.boxplot([scaled_entropy], by = [interruption.split('_cases')[0]],
                         figsize = (4, 6),
                         showmeans = True,
                         notch = True)
    boxplot.set_xlabel("Non-routine transition ("+interruption.split('_cases')[0]+")")
    boxplot.set_ylabel("Entropy")
    boxplot.set_title('')
    boxplot.get_figure()._suptitle.set_visible(False)
    boxplot.get_figure()._suptitle.set_in_layout(False)
    boxplot.xaxis.label.set_size(12)
    boxplot.yaxis.label.set_size(12)
    if save:
        boxplot.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/low_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_boxplot_"+today_date+".png"), bbox_inches='tight')
        boxplot.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_boxplot_"+today_date+".eps"), bbox_inches='tight')
        boxplot.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_boxplot_"+today_date+".tif"), bbox_inches='tight')

    return

def test_LME_assumptions(results, scaled_entropy, job, interruption,  proj_dir, save):
    # Testing the assumptions of using Linear Mixed Effects Regression test
    
    # Test Normality of residual distribution
    ## KDE Plot
    fig = plt.figure(figsize = (6, 4))
    ax = sns.distplot(results.resid, hist = False, kde_kws = {"shade" : True, "lw": 1}, fit = stats.norm)
    ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
    ax.set_xlabel("Residuals")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    # plt.show()

    if save:
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/low_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_KDEplot_"+today_date+".png"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_KDEplot_"+today_date+".eps"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_KDEplot_"+today_date+".tif"), bbox_inches='tight')

    ## Q-Q Plot
    fig = plt.figure(figsize = (6, 4))
    ax = fig.add_subplot(111)
    sm.qqplot(results.resid, dist = stats.norm, line = 's', ax = ax)
    ax.set_title("Q-Q Plot")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    # plt.show()

    if save:
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/low_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_QQplot_"+today_date+".png"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_QQplot_"+today_date+".eps"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_QQplot_"+today_date+".tif"), bbox_inches='tight')

    ## Shapiro-Wilk test
    norm_res = shapiro(results.resid)
    print(f"Testing normality of residuals (Shapiro-Wilk test): {norm_res}")
    
    # Test Independence of errors & Equal variance of errors
    ## RVF plot
    fig = plt.figure(figsize = (6, 4))
    ax = sns.scatterplot(y = results.resid, x = results.fittedvalues)
    ax.set_title("RVF Plot")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    # plt.show()

    if save:
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/low_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_RVFplot_"+today_date+".png"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_RVFplot_"+today_date+".eps"), bbox_inches='tight')
        ax.figure.savefig(os.path.join(proj_dir, "analysis/entropy_validation/plots/high_res/"+scaled_entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LMEresidual_RVFplot_"+today_date+".tif"), bbox_inches='tight')

    return

def save_input_data(df,job,interruption):
    df.to_csv(os.path.join(proj_dir, "processed_data/"+job+"_"+interruption.split('_cases')[0]+"_LME_model_input_"+today_date+".csv"))

def standardize_entropy(df, robust=False):
    if robust:
        # performing robust feature standardization
        '''
        Removes median and scales according to interquartile range IQR. Scales median to 0 & IQR to 1.
            x' = (x-median)/IQR
        Reduces the influence of outliers in the mean/variance estimation, because median and IQR are robust to outliers.
        '''
        print('Performing robust standardization...')
        scaler = preprocessing.RobustScaler().fit(df['entropy'].values.reshape(-1, 1))
        # median
        median = scaler.center_
        print(f'Median: {median}')
        # scaled IQR
        scaled_IQR = scaler.scale_
        print(f'Scaled IQR: {scaled_IQR}')
        
        standardized_entropy = scaler.transform(df['entropy'].values.reshape(-1, 1))
        return [standardized_entropy, median, scaled_IQR]
    else:
        # perform regular standardization
        '''
        Scales features to a mean of 0 and standard deviation of 1
            x' = (x-mean)/std_dev
        '''
        print('Performing regular standardization...')
        scaler = preprocessing.StandardScaler().fit(df['entropy'].values.reshape(-1, 1))
        # mean
        mean = scaler.mean_
        print(f'Mean: {mean}')
        # std_dev
        std_dev = scaler.scale_
        print(f'Std. dev: {std_dev}')
        
        standardized_entropy = scaler.transform(df['entropy'].values.reshape(-1, 1))
        return [standardized_entropy, mean, std_dev]

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
    parser.add_argument('--cache_data', help="Options:True, False | Default=False", default=False)
    parser.add_argument('--save_input_only', help='Options: True, False | Default=False', default=False)
    parser.add_argument('--output_scaling', help='Options: standardization, robust_standardization', default=False)
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
    else:
        cache_data = False

       
    if args.save_input_only == 'True':
        save_input_only = True
    else:
        save_input_only = False

    if args.output_scaling == 'standardization':
        entropy = 'standardized_entropy'
    elif args.output_scaling == 'robust_standardization':
        entropy = 'robust_standardized_entropy'
    else:
        print('No valid scaling method specified. Proceeding with no scaling of the output variable.')
        # no scaling to the output variable (entropy)
        entropy = 'entropy'

    today_date = str(datetime.date.today())
    
    for job in ['Attending', 'APP']:
        print(f"+++++++++++++++++++ Processing {job}s... +++++++++++++++++++")
        for interruption in ['pat_switch_cases_controls', 'to_IB_cases_controls', 'from_IB_cases_controls']:
            print(f"\n++Testing {interruption}...")
            df = pd.read_csv(os.path.join(proj_dir, "processed_data/"+job+'_'+interruption+".csv"))
            # make sure all USER_IDs are strings...
            df['USER_ID'] = df['USER_ID'].astype("string")
            
            # Convert some columns to categorical data type
            df_input = df[['Case', 'timedelta_of_transition_s', 'n_uniq_PAT_ID', 'workShift_date', 'USER_ID', 'H_METRIC_NAME|REPORT_NAME']]\
                        .rename(columns={'Case':interruption.split('_cases')[0],
                                         'H_METRIC_NAME|REPORT_NAME':'entropy',
                                         'n_uniq_PAT_ID':'n_uniq_PAT_seen'
                                        })
            df_input = df_input.astype({"USER_ID":'category',
                                        "workShift_date":'category'
                                       })
            # perform standardization on the entropy values for this population group (job,transition type)
            standardization_output = standardize_entropy(df_input, robust=False)
            df_input['standardized_entropy'] = standardization_output[0]
            standardization_output = standardize_entropy(df_input, robust=True)
            df_input['robust_standardized_entropy'] = standardization_output[0]
            
            if save_input_only:
                save_input_data(df_input, job, interruption)
                continue


            # for entropy in ['entropy', 'standardized_entropy', 'robust_standardized_entropy']:
            # if entropy in ['entropy', 'standardized_entropy']:
            #     #skip for now, because we already ran it yesterday
            #     continue
                
            draw_boxplot(df_input, entropy, job, interruption, proj_dir, save=cache_data)
            
            # This structure allows us to specify nested clustering levels. We assume that the calendar dates are nested under user_ids.
            var_comp = {'workShift_date': '0 + C(workShift_date)'}
            model = sm.MixedLM.from_formula(entropy+' ~ '+interruption.split('_cases')[0]+' + timedelta_of_transition_s + n_uniq_PAT_seen', vc_formula=var_comp, re_formula='1', groups='USER_ID', data=df_input)
            results = model.fit()
            
            # display(results.summary())
            if cache_data:
                savable_df = pd.concat([results.summary().tables[1], results.summary().tables[0]], axis=0)
                savable_df.to_csv(os.path.join(proj_dir, "analysis/entropy_validation/tables/"+entropy+"/"+job+"_"+interruption.split('_cases')[0]+"_LME_model_results_"+today_date+".csv"))
            # print(f"Intraclass Correlation Coefficient (ICC): {float(results.summary().tables[1].loc['USER_ID Var', 'Coef.']) / float(results.summary().tables[0].iloc[2,3])}")
            '''
            Note:
                The notation "Var" after the two clustering variables (eg, USER_ID, ACCESS_DATE) means "Variance".
                For example,
                    The "USER_ID Var" is the random effects of the cluster variable.
                    This models the variation that is present between the USERs. 
                    One can convert variance to standard deviation by taking the square root, this means that on average the USER_ID can vary about sqrt(USER_ID Var)
                    - not much meaning since our clustering variables are categorical
            '''
    
            # Check LME model assumptions
            test_LME_assumptions(results, entropy, job, interruption, proj_dir, save=cache_data)
            

