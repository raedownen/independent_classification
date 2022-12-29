# Import Libraries
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from env import user, password, host
from scipy import stats
from scipy.stats import levene, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import math
import numpy as np
import os
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


###################################################################################
#################################### ACQUIRE DATA #################################
###################################################################################




###################################################################################
##################################### PREP DATA ###################################
###################################################################################
def prep22(df):
    global df22
    df22['charter_encoded'] = df22.charter_status.map({'OPEN ENROLLMENT CHARTER': 1, 'TRADITIONAL ISD/CSD':0})
    df22=df22[(df22.heading == 'A01') | (df22.heading ==  'A03')]
    df22=df22[df22['student_count'] != '-999']
    df22['student_count']= df22['student_count'].str.replace("<", "")
    df22['student_count'] = df22['student_count'].astype(float)
    dfpivot=df22.pivot(index='campus_number', columns='heading', values= 'student_count').dropna()
    df22=df22.merge(dfpivot,how= 'right', on= 'campus_number')
    df22=df22.rename(columns={'A01': 'student_enrollment', 'A03':'discipline_count'})
    df22['discipline_percent']= ((df22['discipline_count']/df22['student_enrollment'])*100)
    df22=df22.round({'discipline_percent': 0})
    df22=df22.drop(columns=['agg_level', 'campus_number', 'region', 'charter_status', 'dist_name_num',                                                       'student_count','section', 'heading', 'heading_name', 'student_count'])
    df22=df22.drop_duplicates()
    df22=df22.reset_index(drop=True)

    #call function with: prep22(df22)

def df_combine(a,b,c,d,e):
    df=pd.concat([df18,df19,df20,df21,df22], ignore_index=True)
    return(df)

#call function with: df_combine(df18,df19,df20,df21,df22)


###################################################################################
#################################### SPLIT DATA ###################################
###################################################################################
#Step 5: Test and train dataset split
def split_tea_data(df):
    '''
    This function performs split on tea data, stratify charter_encoded.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.charter_encoded)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.charter_encoded)
    return train, validate, test

#train, validate, test= split_tea_data(df) 