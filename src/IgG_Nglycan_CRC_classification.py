"""
Python script producing the results presented in scientific paper:
    
    A machine learning framework to extract the biomarker potential
    of IgG N-glycans towards disease risk stratification for targeted screening
    and early diagnosis 

Script authors: Konstantinos Flevaris*, Joseph Davies, Shoh Nakai
@Kontoravdi Lab

*Email: k.flevaris21@imperial.ac.uk
"""

# Basic python:
import os
import warnings
import datetime
import random
from copy import deepcopy

# Non machine learning packages:
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import pingouin as pg
import natsort
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact, wasserstein_distance, norm
from statsmodels.stats.multitest import multipletests
from joblib import dump, load


# Machine learning packages:
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from VennABERS import ScoresToMultiProbs
from alibi.explainers import ALE
from alibi.explainers.ale import plot_ale


# Specify number of cores for computational tasks with sklearn (n_jobs)
# In this particular application, all available cores are used
# -> please set this to a value that is reasonable for your computational resources
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
num_cores = os.cpu_count()
n_jobs_opt = 2
print('Number of available CPU cores:', num_cores)
print('Number of CPU cores used for computational tasks:', n_jobs_opt)

# Seed everything for reproducibility
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

'''
COLORECTAL CANCER (CRC) IgG N-GLYCAN DATASET
'''
filename_dataset = 'CRC_IgG_Nglycan_dataset.xlsx'
df = pd.read_excel(filename_dataset)

# Ensure 'Age' is float
df['Age'] = df['Age'].astype('float64')

# Extract year from 'sample.date'
df['Sample.date'] = pd.to_datetime(df['Sample.date'])
df['Year'] = df['Sample.date'].dt.year

# Remove specific columns that won't be used for modelling
df = df.drop(columns=['Unnamed: 0', 'Sample.ID', 'Sample.date', 'BMI'])

# Create 'Year_Plate' column and drop 'Year' and 'Plate'
df.insert(2, 'Year_Plate', df['Year'].astype(str) + '_' + df['Plate'].astype(str))
df = df.drop(columns=['Year', 'Plate'])

# Identify unique 'Year_Plate' and drop corresponding rows
unique_year_plate = df['Year_Plate'].value_counts() == 1
df = df[~df['Year_Plate'].isin(unique_year_plate[unique_year_plate].index)]

# Reset index
df = df.reset_index(drop=True)

# Store column data types and dataset
df_dtypes = df.dtypes


'''
EXPLORATORY DATA ANALYSIS
'''
def bar_chart_of_age_and_cancer_status(df):
    '''
    Creates bar chart to show how many cancer and control samples the data has at different ages
    
    Parameters:
        df (pd.DataFrame): Dataframe input data, with the columns 'Age' and 'Status' or 'Status_Cancer'
        
    Returns:
        Frequency-Age bar chart with legend values for control and cancer patients
    '''
    
    # Bar chart plotting:
    if 'Status_Cancer' in df.columns:
        pd.crosstab(df['Age'], df['Status_Cancer']).plot(kind='bar',figsize=(20,6))
    elif 'Status' in df.columns:
        pd.crosstab(df['Age'], df['Status']).plot(kind='bar',figsize=(20,6))
        
    # Formatting:
    plt.title('Cancer Frequency for Different Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    plt.close()
    
'''
PREPROCESSING
'''    
def encode(df):
    '''
    Returns a dataframe with binary 1's and 0's for gender and cancer status columns    
    
    Parameters:
        df (pd.DataFrame): Dataframe containing the column 'Sex' with either 'M' or 'F' entries, and 'Cancer' or 'Control' for the column 'Status' 
        
    Returns:
        df (pd.DataFrame): Dataframe containing the column 'Sex_M' with either 1 for males or 0 for females, and the column 'Status_Cancer' with 1 for cancer patients and 0 for controls
    '''
    # Encode Gender and (Cancer) Status:
    df = pd.get_dummies(df, columns=['Sex', 'Status'])
    df = df.drop(columns=['Sex_F', 'Status_Control'])
    
    # Add in new columns using list notation:
    cols = list(df.columns.values)
    cols.pop(cols.index('Sex_M'))
    cols.pop(cols.index('Status_Cancer'))
    
    # Compile into one dataframe
    df = df[['Sex_M'] + cols + ['Status_Cancer']]
    return df    

def ComBat_correction_GP(df_encoded_log, covariates, batch_effects, consider_covariates):
    '''
    Corrects the batch effects in the data using the ComBat method.

    Parameters:
        df_encoded_log (pd.DataFrame): The input dataframe, which contains covariates, features, and class labels.
        covariates (list): A list of column names in the dataframe that represent the covariates.
        batch_effects (list): A list of column names in the dataframe that represent the batch effects.

    Returns:
        df_encoded_log_corr (pd.DataFrame): The corrected dataframe where batch effects have been minimized.
    '''
    
    # Select columns related to GPs
    GP_mask = df_encoded_log.columns.str.contains('GP')
    
    # Get the data related to GPs and other data
    GP_data = df_encoded_log.loc[:, GP_mask]
    GP_names = GP_data.columns.tolist()
    rest_data = df_encoded_log.loc[:, ~GP_mask]
    
    # Create an AnnData object for the GP data and other attributes
    GP_anndata = anndata.AnnData(X=GP_data.values,
                                 obs=df_encoded_log[covariates + batch_effects].copy(),
                                 var=pd.DataFrame(index=GP_data.columns))

    # Convert the covariates and batch effects columns to category codes if they are not floats
    for column in covariates + batch_effects:
        if GP_anndata.obs[column].dtype != np.float64:
            GP_anndata.obs[column] = GP_anndata.obs[column].astype('category').cat.codes
    
    # Apply ComBat correction for batch effects
    for batch in batch_effects:
        if consider_covariates:
            GP_anndata_corr = sc.pp.combat(GP_anndata, key=batch, covariates=covariates, inplace=False)
        else:
            GP_anndata_corr = sc.pp.combat(GP_anndata, key=batch, inplace=False)
    
    # Convert the corrected AnnData back to DataFrame
    GP_data_corr = pd.DataFrame(GP_anndata_corr, columns=GP_names)
    
    # Concatenate the corrected GP data with the rest of the data
    df_encoded_log_corr = pd.concat([rest_data, GP_data_corr], axis=1).drop(columns=batch_effects)
    
    # Reorder the columns to move 'Status_Cancer' to the end
    df_encoded_log_corr = df_encoded_log_corr.assign(Status_Cancer=df_encoded_log_corr.pop('Status_Cancer'))

    return df_encoded_log_corr


def split_df_to_X_and_y(df):
    '''
    Converts a single dataframe into two smaller dataframes: X (data the model uses to make predictions) and y (status cancer)
    
    Parameters:
        df (pd.DataFrame): Dataframe for the entire dataset
        
    Returns:
        X (pd.DataFrame): Dataframe of all columns WITHOUT 'Status_Cancer' column
        y (pd.Series): Series of ONLY Status_Cancer' column
    '''
     
    X = df.drop(columns=['Status_Cancer'])    
    y = df['Status_Cancer'].astype(int)
    return X, y


'''
HYPOTHESIS TESTING AND DATA AUGMENTATION
'''
def smote_data_augmentation(df_train_encoded):
    '''
    Augments (creates synthetic control patient data) for patients aged between 61 and 82, using a SMOTE (synthetic minority over-sampling technique) from the few existing controls
    
    Parameters:
        df_train_encoded (pd.DataFrame): Original dataframe with no augmented/ synthetic data
        
    Returns:
        df_data_aug_storage (pd.DataFrame): Dataframe with augmented data for control patients up to age 82
    '''
    # Suppress warnings:
    warnings.filterwarnings('ignore')
    
    # Extract train and test data:
    df_train_continuous = df_train_encoded.drop(columns=['Sex_M', 'Age'])
    X_train_smote = df_train_continuous.drop(columns=['Status_Cancer'])
    y_train_smote = df_train_continuous['Status_Cancer']

    # Perform oversampling:
    ages = np.linspace(61, 82, 82-60)

    # Create storage DataFrame for augmented data:
    df_smote_aug = pd.DataFrame(columns = df_train_encoded.columns)
    
    for age in ages: 
        # Oversampling for GPs:
        len_difference = len(df_train_encoded.loc[(df_train_encoded['Age'] == age) & (df_train_encoded['Status_Cancer'] == 1)]) \
                        - len(df_train_encoded.loc[(df_train_encoded['Age'] == age) & (df_train_encoded['Status_Cancer'] == 0)])  
        sampling_strategy = {0: len_difference + 538}
        oversample = SMOTE(random_state=SEED, k_neighbors=3, sampling_strategy=sampling_strategy)
        X_res, y_res = oversample.fit_resample(X_train_smote, y_train_smote)
        X_res = X_res.tail(len_difference).reset_index(drop=True)

        # All data from oversampling are for control (ie. no cancer):
        listofzeros = [0] * len_difference 
        y_res = pd.DataFrame(listofzeros, columns=['Status_Cancer'])

        # Assignment of Gender_M is random:
        gender_m = [random.randint(0, 1)] * len_difference
        X_res_gender_m = pd.DataFrame(gender_m, columns = ['Sex_M'])

        # Assignment of age:
        age = [age] * len_difference
        X_res_age = pd.DataFrame(age, columns = ['Age'])

        # Concatenate all data: 
        aug_data = pd.concat([X_res_gender_m, X_res_age, X_res, y_res], axis = 1)
        df_smote_aug = pd.concat([df_smote_aug, aug_data], ignore_index=True)
    return df_smote_aug


def smote_age_range_data_augmentation(df_train_encoded):
    '''
    Augments (creates synthetic control patient data) for patients aged between 61 and 74, using a SMOTE (synthetic minority over-sampling technique) from the few existing controls
    
    Parameters:
        df_train_encoded (pd.DataFrame): Original dataframe with no augmented/ synthetic data
        
    Returns:
        df_data_aug_storage (pd.DataFrame): Dataframe with augmented data for control patients up to age 74
    '''
    warnings.filterwarnings('ignore')
    
    # Obtain X_train and y_train after dropping Gender_M and Age:
    df_train_74 = df_train_encoded.drop(df_train_encoded.loc[df_train_encoded['Age'] > 74].index)
    df_train_continuous = df_train_74.drop(['Sex_M', 'Age'], axis=1)
    X_train_SMOTE = df_train_continuous.drop(['Status_Cancer'], axis=1)
    y_train_SMOTE = df_train_continuous['Status_Cancer']

    # Perform oversampling in desired age range:
    ages = np.linspace(61, 74, 74-61+1)

    # Create storage df for augmented data
    SMOTE_aug = pd.DataFrame(columns = df_train_continuous.columns)
    age_list = []

    for i in ages: 
        # Set the len_difference (attempt to make the median age the same across cancer and control):
        if i < 72:
            len_difference = len(df_train_74.loc[(df_train_74['Age'] == i) & (df_train_74['Status_Cancer'] == 1)]) \
                            - len(df_train_74.loc[(df_train_74['Age'] == i) & (df_train_74['Status_Cancer'] == 0)])      
        else:
            len_difference = round((len(df_train_74.loc[(df_train_74['Age'] == i) & (df_train_74['Status_Cancer'] == 1)]) \
                            - len(df_train_74.loc[(df_train_74['Age'] == i) & (df_train_74['Status_Cancer'] == 0)]))/2, 0)
            len_difference = int(len_difference)
            
        # Oversampling for GPs:
        sampling_strategy = {0: len_difference + len(df_train_encoded)}
        oversample = SMOTE(random_state=SEED, k_neighbors=5, sampling_strategy=sampling_strategy)
        X_res, y_res = oversample.fit_resample(X_train_SMOTE, y_train_SMOTE)
        X_res = X_res.tail(len_difference).reset_index(drop=True)

        # All data from oversampling are for control (ie. no cancer):
        listofzeros = [0.0] * len_difference 
        y_res = pd.DataFrame(listofzeros, columns=['Status_Cancer'])

        # Assignment of age :
        for j in range(len_difference):
            age_list.append(i)

        # Concatenate GP features and label:
        aug_data = pd.concat([X_res, y_res], axis = 1)
        SMOTE_aug = pd.concat([SMOTE_aug, aug_data], ignore_index=True)

    # Add gender data:
    gender_list = []
    for i in range(len(SMOTE_aug)):
        np.random.seed(0)
        random_int = random.randint(1, 7)
        # NOTE- could do this better by using exact ratio of males to females, instead of approximate 4:3 ratio used here
        if random_int == 1 or random_int == 2 or random_int == 3 or random_int == 4:
            gender = 1
        if random_int == 5 or random_int == 6 or random_int == 7:
            gender = 0

        gender_list.append(gender)
    SMOTE_aug.insert(0, 'Sex_M', gender_list)

    # Add age data 
    SMOTE_aug.insert(1, 'Age', age_list)
    SMOTE_aug

    return SMOTE_aug


def data_augmentation(df_train_encoded, method='SMOTE'):
    '''
    Generates an augmented binary dataset using SMOTE and normal distributions, both with and without extrapolating beyond the oldest control patient
    
    Parameters:
        df_train_encoded (pd.DataFrame): Original dataframe with no augmented/ synthetic data
        method (str): Desired method to augment data with, OPTIONS: 'SMOTE', 'Normal', 'Normal with Age Range', 'SMOTE with Age Range'
        
    Returns:
        df_data_aug_storage (pd.DataFrame):  Dataframe with augmented data for control patients up to desired age
    '''
    
    # Choose augmentation method:
    if method == 'SMOTE':
        df_data_aug = smote_data_augmentation(df_train_encoded)
        df_train_augmented_merged = pd.concat([df_train_encoded, df_data_aug])
    elif method == 'SMOTE with Age Range':
        df_data_aug = smote_age_range_data_augmentation(df_train_encoded)
        df_train_74 = df_train_encoded.drop(df_train_encoded.loc[df_train_encoded['Age'] > 74].index)
        df_train_augmented_merged = pd.concat([df_train_74, df_data_aug])
    return df_train_augmented_merged.reset_index(drop=True)


def mann_whitney_u_test(df_encoded, str_augmentation_method='No Augmentation'):
    '''
    Conducts Mann–Whitney U test to determine if there is a statistically signficant difference between cancer and control distributions for a given CONTINUOUS feature (column)
    
    Parameters:
        df_encoded (pd.DataFrame): Dataframe with encoded features (gender and cancer status)
        str_augmentation_method (str): Method used to augment data (title of dataframe)
        
    Returns:
        df_corrected_p_values (pd.Dataframe): A dataframe with corrected p-values of each feature and a binary yes or no determination of whether its signficant at 95% confidence
    '''
    
    # Get cancer features:
    df_cancer = df_encoded.loc[df_encoded['Status_Cancer'] == 1]
    features_cancer = df_cancer.drop(columns=['Status_Cancer', 'Sex_M']).to_dict('list')
    
    # Get control features:
    df_control = df_encoded.loc[df_encoded['Status_Cancer'] == 0]
    features_control = df_control.drop(columns=['Status_Cancer', 'Sex_M']).to_dict('list')
    
    # Intialise variables for test:
    nx, ny = len(df_cancer), len(df_control)
    features_U1 = {}
    features_U2 = {}
    features_p_values = {}
    features_list = []

    # Run test:
    for key in features_cancer:
        features_U1[key], features_p_values[key]  = mannwhitneyu(features_cancer[key], features_control[key], use_continuity=False)
        features_U2[key] = nx*ny - features_U1[key]
        features_list.append(key)
    
    features_p_values_array = np.array(list(features_p_values.values()))
    features_p_values_corrected = multipletests(pvals=features_p_values_array, alpha=0.05, method='fdr_bh') # the function arranges the p-values in ascending order, but returns the original order
    features_p_values_corrected_array = features_p_values_corrected[1]
    features_p_values_corrected_array = np.round(features_p_values_corrected_array, 5)
    
    # Collate into DataFrame:
    df_corrected_p_values = pd.DataFrame(features_p_values_corrected_array, columns=['Corrected p-values']) 
    df_corrected_p_values['Statistically significant difference'] = df_corrected_p_values['Corrected p-values'].apply(
        lambda x: 'Yes' if x < 0.05 else 'No')
    df_corrected_p_values.index = features_list
    df_corrected_p_values.columns = [[str_augmentation_method, str_augmentation_method], df_corrected_p_values.columns]
    return df_corrected_p_values

def fisher_exact_test(df_train_encoded, method, title):
    '''
    Conducts Fisher's exact test to determine if there is a statistically signficant difference between cancer and control distributions for a given CATEGORICAL feature (column)
    
    Parameters:
        df_train_encoded (pd.DataFrame): Dataframe with encoded features (gender and cancer status)
        method (str): Method used to augment data
        title (str): Desired title of dataframe
        
    Returns:
        fisher_stats (pd.Dataframe): A dataframe with corrected p-values of each feature and a binary yes or no determination of whether it is signficant at 95% confidence
    '''
    
    if method == 'No Augmentation':
        str_augmentation_method = 'No Augmentation'
    elif method == 'Normal':
        str_augmentation_method = 'Normal'
    elif method == 'SMOTE':
        str_augmentation_method = 'SMOTE'
    elif method == 'Normal with Age Range':
        str_augmentation_method = 'Normal (61-74)'
    elif method == 'SMOTE with Age Range':
        str_augmentation_method = 'SMOTE (61-74)'

    if method == 'No Augmentation (21-82)' or method == 'No Augmentation (21-60)':
        df_train_augmented_merged = df_train_encoded
    else:
        df_train_augmented_merged = data_augmentation(df_train_encoded, method) 
        
    # Assuming there are more categorical columns to test
    categorical_cols = ['Sex_M']  # Add more categorical columns here

    pvalues = []
    indices = []

    for col in categorical_cols:
        crosstab = pd.crosstab(df_train_augmented_merged['Status_Cancer'], df_train_augmented_merged[col]).to_numpy()
        _, pvalue = fisher_exact(crosstab)
        pvalues.append(pvalue)
        indices.append(col)

    pvalues_array = np.array(pvalues)
    pvalues_corrected = multipletests(pvals=pvalues_array, alpha=0.05, method='fdr_bh')
    pvalues_corrected_array = pvalues_corrected[1]
    pvalues_corrected_array = np.round(pvalues_corrected_array, 5)

    fisher_stats = pd.DataFrame(pvalues_corrected_array, columns=['Corrected p-values'], index=indices)
    fisher_stats['Statistically significant difference'] = fisher_stats['Corrected p-values'].apply(
        lambda x: 'Yes' if x < 0.05 else 'No')
    fisher_stats.columns = [[title, title], fisher_stats.columns]

    return fisher_stats

# Encoding
df_encoded = encode(df)
feature_names = df_encoded.drop(columns=['Status_Cancer']).columns.to_list()

# Prepare data for batch correction using log transformation (only on GP features)
df_encoded_log = df_encoded.copy()
df_encoded_log.update(df_encoded_log.filter(like='GP').apply(np.log))


df_encoded_log_corr = ComBat_correction_GP(df_encoded_log, 
                                            covariates=['Sex_M', 'Age'], 
                                            batch_effects=['Year_Plate'],
                                            consider_covariates=True)

df_encoded_log_corr_exp = df_encoded_log_corr.copy()
df_encoded_log_corr_exp.update(df_encoded_log_corr_exp.filter(like='GP').apply(np.exp))


'''
BINARY CLASSIFICATION DATASET
'''
df_binary = df_encoded_log_corr.copy()

# Separate class labels from features:
y_binary = df_binary['Status_Cancer']
X_binary = df_binary.drop(['Status_Cancer'], axis=1)

# Check how many controls and cancer cases there are:
print('Counts of class labels in raw binary dataset:\n', df_binary['Status_Cancer'].value_counts()) # there is class imbalance (i.e. there are more samples that are labeled with class 'Cancer' that with class 'Control')

# Check cancer frequency for different ages in original dataset
bar_chart_of_age_and_cancer_status(df_binary)  # TLDR: there are almost no controls for older-aged samples

# Concatenate df_train_encoded and data_aug_storage
df_train_merged = data_augmentation(df_binary, method='SMOTE with Age Range')

# Check cancer frequency for different ages in augmented dataset
bar_chart_of_age_and_cancer_status(df_train_merged)

# Compile dictionary for statistical comparison of batch corrected data:
d_non_augmented_dataframes = {
    'No Augmentation (21-82)': df_binary,
    'No Augmentation (21-60)': df_binary.drop(df_binary[df_binary['Age'] > 60].index),
    'SMOTE (21-82)': data_augmentation(df_binary, method='SMOTE'),
    'SMOTE (21-74)': data_augmentation(df_binary, method='SMOTE with Age Range') 
}

# Run Mann–Whitney U test
d_mann_whitney_u_results = {}
for key, val in d_non_augmented_dataframes.items():
    d_mann_whitney_u_results[key] = mann_whitney_u_test(val, key)

# Display results:   
print('Mann–Whitney U test for Age and GPs hypothesis testing at 95% confidence for a statistical difference between control and cancer distributions')
df_mann_whitney_u_results = pd.concat(d_mann_whitney_u_results.values(), axis=1)
df_mann_whitney_u_results.to_excel('mann_whitney_u_results_Age_GPs_binary.xlsx')

# Create dictionary for Fisher's exact test:
d_fisher_methods_and_titles = {
    'No Augmentation (21-82)': 'No Augmentation (21-82)',
    'No Augmentation (21-60)': 'No Augmentation (21-60)',
    'SMOTE': 'SMOTE (21-82)',
    'SMOTE with Age Range': 'SMOTE (21-74)',
}

# Run Fisher exact test
d_fisher_results_training = {}
for key, val in d_fisher_methods_and_titles.items():
    if key == 'No Augmentation (21-60)':
        d_fisher_results_training[key] = fisher_exact_test(df_binary.drop(df_binary[df_binary['Age'] > 60].index), key, val)
    else:
        d_fisher_results_training[key] = fisher_exact_test(df_binary, key, val)
    
# Display results:
print('Fisher exact test for Sex_M hypothesis testing at 95% confidence for a statistical difference between control and cancer distributions')
df_fisher_results = pd.concat(d_fisher_results_training.values(), axis=1)
df_fisher_results.to_excel('fisher_results_SexM_binary.xlsx')

# Non-augmented binary dataset
df_train_non_augmented = df_binary.copy()
df_train_non_augmented = df_train_non_augmented.drop(df_train_non_augmented[df_train_non_augmented['Age'] > 60].index)
df_train_non_augmented_correl = df_train_non_augmented.copy()
df_train_non_augmented = df_train_non_augmented.drop(columns=['Sex_M', 'Age'])
# df_train_non_augmented.to_excel('df_train_non_augmented_binary.xlsx')
X_train_non_augmented, y_train_non_augmented = split_df_to_X_and_y(df_train_non_augmented)

# Check how many controls and cancer cases there are for non-augmented dataset
print('Counts of class labels in non-augmented binary dataset:\n', df_train_non_augmented['Status_Cancer'].value_counts())

# Augmented binary dataset
df_train_smote_augmented = data_augmentation(df_binary, 'SMOTE with Age Range')
df_train_smote_augmented = df_train_smote_augmented.drop(df_train_smote_augmented[df_train_smote_augmented['Age'] > 74].index)
df_train_smote_augmented_correl = df_train_smote_augmented.copy()
df_train_smote_augmented = df_train_smote_augmented.drop(columns=['Sex_M', 'Age'])
df_train_smote_augmented.to_excel('df_train_smote_augmented_binary.xlsx')
X_train_augmented, y_train_augmented = split_df_to_X_and_y(df_train_smote_augmented)

# Check how many controls and cancer cases there are for augmented dataset
print('Counts of class labels in augmented binary dataset:\n', df_train_smote_augmented['Status_Cancer'].value_counts())

# Print the shape of the non_augmented and smote_augmented datasets
print('Non-Augmented training data shape:', df_train_non_augmented.shape)
print('Augmented training data shape:', df_train_smote_augmented.shape)


'''
PAIRWISE CORRELATION ANALYSIS FOR GLYCAN FEATURES OF NON-AUGMENTED AND AUGMENTED BINARY DATASET
'''
def glycan_corr_with_fdr(df, class_label, covariates,  method, feature_names=None, padj_method='fdr_bh', fdr=0.05):
    '''  
    Computes the pairwise partial correlations for glycan features and corrects for multiple testing using the specified method, such as Benjamini-Hochberg, to control the False Discovery Rate (FDR).

    Parameters:
        df (pd.DataFrame or array-like): The input data, which can either be a Pandas DataFrame or an array-like structure. If it's not a DataFrame, the `feature_names` parameter must be provided.
        class_label (str or None): Specifies the class label for filtering the data based on the 'Status_Cancer' column. If set to `None`, no filtering is done.
        covariates (list): A list of column names in the DataFrame that represent the covariates to be adjusted for in the partial correlation computation.
        method (str): The correlation method to be used (e.g., 'spearman', 'pearson', etc.).
        feature_names (list, optional): A list of feature names to be used if `df` is not a DataFrame.
        padj_method (str, default='fdr_bh'): The method used for adjusting p-values for multiple comparisons. Default is Benjamini-Hochberg ('fdr_bh').
        fdr (float, default=0.05): The alpha level for controlling the False Discovery Rate.

    Returns:
        pairwise_partial_corr (pd.DataFrame): A DataFrame containing the pairwise partial correlations along with p-values, adjusted p-values, and a significance flag.
        rho (pd.DataFrame): A square DataFrame containing the pairwise partial correlations arranged in a matrix format, with non-significant correlations set to zero.
    '''
    
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=feature_names)
        
    # Choose feature values based on class label
    if class_label != None:
        df = df[df['Status_Cancer'] == class_label]
    else:
        pass
    
    # Specify the glycan feature columns
    df.columns = df.columns.astype(str)
    GP_mask = df.columns.str.contains('GP')
    features_df_GP = df.loc[:, GP_mask]
    feature_columns_GP = features_df_GP.columns.tolist()
    
    # Compute pairwise partial correlations using the Spearman method
    pairwise_partial_corr = pg.pairwise_corr(df, columns=feature_columns_GP, covar=covariates, method=method)
        
    # Correct for multiple testing using the Benjamini-Hochberg method with FDR = 0.05
    reject, adj_p_values, _, _ = multipletests(pairwise_partial_corr['p-unc'].values, alpha=fdr, method=padj_method)
    
    # Add the adjusted p-values and significance flag to the results DataFrame
    pairwise_partial_corr['p-adj'] = adj_p_values
    pairwise_partial_corr['Significant'] = reject
    
    # Set the partial correlation coefficient to zero for non-significant pairs
    pairwise_partial_corr.loc[pairwise_partial_corr['Significant'] == False, 'r'] = 0

    # Create a square DataFrame filled with NaNs
    rho = pd.DataFrame(index=feature_columns_GP, columns=feature_columns_GP)

    # Fill the rho DataFrame with the partial correlation values from the results DataFrame
    for i, row in pairwise_partial_corr.iterrows():
        x, y, r = row['X'], row['Y'], row['r']
        rho.at[x, y] = r
        rho.at[y, x] = r

    # Fill the diagonal with ones
    np.fill_diagonal(rho.values, 1)

    # Change all data types to float
    rho = rho.astype(float)

    # Apply natural sorting to row and column names
    rho = rho.sort_index(axis=0, key=natsort.natsort_keygen())
    rho = rho.sort_index(axis=1, key=natsort.natsort_keygen())
    
    return pairwise_partial_corr, rho

def visualize_glycan_correlations(rho, title_corr='', num_clusters=4):
    
    # Create the correlation heatmap
    plt.figure(figsize=(16,5))
    lower_rho = rho.where(np.tril(np.ones(rho.shape), k=0).astype(bool))
    heatmap = sns.heatmap(lower_rho.round(2), vmin=-1, vmax=1, annot=True, mask=lower_rho.isna())
    # heatmap.set_title('Partial Spearman Correlation Heatmap of Features', fontdict={'fontsize':12}, pad=12)
    plt.savefig(title_corr + ' Correlation Heatmap.png',
                format='png',
                bbox_inches='tight',
                orientation='portrait',
                dpi=1200)
    plt.show()


# The correlation structure for both non-augmented and augmented training datasets is almost identical

# Non-augmented
p_corr_non_aug, rho_non_aug = glycan_corr_with_fdr(df_train_non_augmented_correl, 
                                                    class_label=None, 
                                                    covariates=None,
                                                    method='pearson')
high_corr_non_aug = p_corr_non_aug[np.abs(p_corr_non_aug['r']) > 0.7]

# Augmented
p_corr_aug, rho_aug = glycan_corr_with_fdr(df_train_smote_augmented_correl, 
                                            class_label=None,
                                            covariates=None,
                                            method='pearson')
high_corr_aug = p_corr_aug[np.abs(p_corr_aug['r']) > 0.7]

# Augmented
p_corr_aug_cov, rho_aug_cov = glycan_corr_with_fdr(df_train_smote_augmented_correl, 
                                            class_label=None,
                                            covariates=['Age', 'Sex_M'],
                                            method='pearson')
high_corr_aug_cov = p_corr_aug_cov[np.abs(p_corr_aug_cov['r']) > 0.7]
visualize_glycan_correlations(rho_aug_cov, title_corr='Augmented')
 
def all_scalers(scaling_methods):
    '''
    Returns python dictionary of scaler objects used to scale X data
    
    Paramaters: 
        scaling_methods (list of str): Desired scaling methods ('Min Max', 'Standard', 'Robust', or 'None')
        
    Returns:
        scalers_dict: Python dictionary of scaling objects
    '''
    
    # define all possible scalers
    all_possible_scalers = {
        'Min Max': MinMaxScaler(),
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
        'None': None   
    }

    # make sure the input is a list
    if not isinstance(scaling_methods, list):
        raise TypeError('scaling_methods must be a list')
    
    # make sure the list is not empty
    if not scaling_methods:
        raise ValueError('scaling_methods must not be empty')
        
    scalers_dict = {}
    for scaling_method in scaling_methods:
        if scaling_method in all_possible_scalers:
            scalers_dict[scaling_method] = all_possible_scalers[scaling_method]
        else:
            raise ValueError(f'Unknown scaling method: {scaling_method}')
    
    return scalers_dict

'''
FEATURE IMPORTANCE ANALYSIS
'''

'''
PROBABILITY CALIBRATION
'''
class CalibratedPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline
        self.is_fitted = False
        self.calibrPoints = None

    def fit(self, X, y, refit_estimator=True):
        if refit_estimator:
            self.base_pipeline.fit(X, y)
        y_pred_proba = self.base_pipeline.predict_proba(X)
        self.calibrPoints = list(zip(y_pred_proba[:, 1], y))  # Store scores and true labels for the positive class
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise NotFittedError('The pipeline is not calibrated.')
        y_pred_proba = self.base_pipeline.predict_proba(X)[:, 1]  # Only consider the positive class
        p0, p1 = ScoresToMultiProbs(self.calibrPoints, y_pred_proba)  # Calibrate the scores
        p = p1 / (1 - p0 + p1)
        y_pred_calibrated = np.vstack((1 - p, p)).T  # Stack the probabilities for the negative and positive class
        return y_pred_calibrated

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)  # Return the class with the highest probability
    
def ECE(true_labels, predictions, n_bins=10):
    '''
    Computes the Expected Calibration Error (ECE)

    Parameters:
        true_labels (np.ndarray): True labels
        predictions (np.ndarray): Pipeline output probabilities
        n_bins (int): Number of bins to use

    Returns:
        ece (float): The expected calibration error
    '''
    # Convert inputs to numpy arrays if they aren't already
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
        
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)

    if len(predictions.shape) == 1:
        predictions = np.vstack((1 - predictions, predictions)).T  # Shape: (n_samples, 2)
        
    pred_confidences = np.max(predictions, axis=1)

    bin_limits = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for bin_min, bin_max in zip(bin_limits[:-1], bin_limits[1:]):
        in_bin = np.logical_and(pred_confidences >= bin_min, pred_confidences < bin_max)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(true_labels[in_bin] == np.argmax(predictions[in_bin], axis=1))
            avg_confidence_in_bin = np.mean(pred_confidences[in_bin])
            
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece
    

def plot_reliability_diagram(calibration, calibration_info_uncalib, calibration_info_calib, dataset):
    '''
    Plots a reliability diagram to visually assess the calibration performance of a classification pipeline, both before and after calibration
    It provides the Expected Calibration Error (ECE) and Log Loss for each case

    Parameters:
        calibration (bool): A flag indicating whether or not to include the calibrated pipeline data in the plot
        calibration_info_uncalib (tuple): A tuple containing four elements: 
            1. `fraction_of_positives_uncalib`: Fraction of true positives for each bin in uncalibrated data
            2. `mean_predicted_value_uncalib`: Mean predicted probabilities for each bin in uncalibrated data
            3. `ECE_score_uncalib`: The Expected Calibration Error score for uncalibrated data
            4. `log_loss_uncalib`: The Log Loss for uncalibrated data
        calibration_info_calib (tuple): A tuple containing four elements, similar to `calibration_info_uncalib`, but for calibrated data:
            1. `fraction_of_positives_calib`: Fraction of true positives for each bin in calibrated data
            2. `mean_predicted_value_calib`: Mean predicted probabilities for each bin in calibrated data
            3. `ECE_score_calib`: The Expected Calibration Error score for calibrated data
            4. `log_loss_calib`: The Log Loss for calibrated data
        dataset (str): The name of the dataset being used, which will appear in the plot title

    Returns:
        fig (matplotlib.figure.Figure): A Matplotlib Figure object containing the reliability diagram

    '''
    fig, ax = plt.subplots(figsize=(7, 7))
    
    fraction_of_positives_uncalib, mean_predicted_value_uncalib, ECE_score_uncalib, log_loss_uncalib = calibration_info_uncalib
    fraction_of_positives_calib, mean_predicted_value_calib, ECE_score_calib, log_loss_calib = calibration_info_calib

    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax.plot(mean_predicted_value_uncalib, fraction_of_positives_uncalib, 's-', color='blue',
            label='%s (ECE=%.3f / Log Loss=%.3f)' % ('Uncalibrated Pipeline', ECE_score_uncalib, log_loss_uncalib))
    if calibration:
        ax.plot(mean_predicted_value_calib, fraction_of_positives_calib, 's-', color='orange',
                label='%s (ECE=%.3f / Log Loss=%.3f)' % ('Calibrated Pipeline', ECE_score_calib, log_loss_calib))
    else:
        pass
    
    ax.set_ylabel('Fraction of positives (Positive class: Cancer)', fontsize='medium')
    ax.set_xlabel('Mean predicted value (Positive class: Cancer)', fontsize='medium')
    ax.set_title('Reliability Diagram for Pipeline (' + dataset + ' dataset)', fontsize='large')
    
    ax.legend(loc='best', fontsize='medium')
    ax.grid(True)
    
    fig.tight_layout()
    return fig

'''
TRAINING, TUNING, EVALUATION
'''

def plot_roc_auc_curve_binary(roc_curve_calib, dataset):
    '''
    Plots mean AUC-ROC curve with ±1 standard error for binary calibrated tuned pipeline
    
    Parameters:
        roc_curve_calib (dict): Dictionary of all AUC-ROC data across all outer loop folds for calibrated tuned pipeline
        dataset (str): Type of dataset ('Non-Augmented', 'Augmented')
    Returns:
        fig (matplotlib.figure.Figure): A Matplotlib Figure object containing the mean AUC-ROC curve with ±1 standard error

    '''
    
    fig, ax = plt.subplots(figsize=(6, 6))

    # Initialize arrays for storing the TPRs and FPRs for each fold
    tprs = []
    fprs = np.linspace(0, 1, 100)

    # Calculate mean TPR and AUC for each FPR value
    for key, value in roc_curve_calib.items():
        if 'Outer Loop Fold' in key:
            fpr_calib = value[0]
            tpr_calib = value[1]
            tpr = np.interp(fprs, fpr_calib, tpr_calib)
            tprs.append(tpr)

    mean_tpr = np.mean(tprs, axis=0)
    se_tpr = np.std(tprs, axis=0) / np.sqrt(len(tprs)) # calculate standard error
    mean_auc = np.trapz(mean_tpr, fprs)

    # Plot mean ROC curve
    plt.plot(fprs, mean_tpr, label='Calibrated Pipeline (AUC = {:.3f})'.format(mean_auc), color='blue', lw=2)

    # Plot 1 standard deviation around mean ROC curve
    tprs_upper = np.minimum(mean_tpr + se_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - se_tpr, 0)
    ax.fill_between(fprs, tprs_lower, tprs_upper, color='skyblue', alpha=0.3, label='± 1 SE')

    # Plot random guess curve
    ax.plot([0, 1], [0, 1], linestyle='--', c='black', label='Random Guess (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize='medium')
    ax.set_ylabel('True Positive Rate', fontsize='medium')
    ax.set_title('ROC Curve for Pipeline (' + dataset + ' dataset)', fontsize='large')
    
    ax.legend(loc='lower right', fontsize='medium')
    ax.grid(True)
    
    fig.tight_layout()
    return fig


def all_raw_clfs(classifiers):
    '''
    Selects correct python ML model given its name
    
    Paramaters:
        classifiers (list of str): name of ML model to be used ('LR', 'SVM', 'RF' or 'XGB')
    
    Returns:
        clfs_dict: python dictionary corresponding to the given machine learning models
    '''
    
    # define all possible classifiers
    all_possible_clfs = {
        'LR': LogisticRegression(random_state=SEED, max_iter=100000, n_jobs=n_jobs_opt),
        'SVM': SVC(random_state=SEED, max_iter=100000, probability=True),
        'RF': RandomForestClassifier(random_state=SEED, n_jobs=n_jobs_opt),
        'XGB': XGBClassifier(random_state=SEED, max_iter=10000, n_jobs=n_jobs_opt, verbosity=0, objective='binary:logistic', eval_metric='auc') 
    }
        
    # make sure the input is a list
    if not isinstance(classifiers, list):
        raise TypeError('classifiers must be a list')
    
    # make sure the list is not empty
    if not classifiers:
        raise ValueError('classifiers must not be empty')
        
    clfs_dict = {}
    for classifier in classifiers:
        if classifier in all_possible_clfs:
            clfs_dict[classifier] = all_possible_clfs[classifier]
        else:
            raise ValueError(f'Unknown classifier: {classifier}')
    
    return clfs_dict

def get_hyperparam_space(trial, clf_name):
    '''
    Defines the hyperparameter search space for different classifiers (Logistic Regression, Support Vector Machines, Random Forest, and XGBoost). It is designed to work in conjunction with Optuna trials for hyperparameter optimization

    Parameters:    
        trial (Optuna.trial.Trial): An Optuna trial object for hyperparameter optimization
        clf_name (str): The name of the classifier for which the hyperparameter space is being defined 
        Must be one of the following:
        - 'LR': Logistic Regression
        - 'SVM': Support Vector Machines
        - 'RF': Random Forest
        - 'XGB': XGBoost

    Returns:
        params (dict): A dictionary containing the hyperparameter search space for the specified classifier The keys correspond to hyperparameter names, and the values specify the ranges or options for each hyperparameter.
    '''
    
    # Define the search space for each classifier (clf__ is required for compatibility with sklearn.pipeline)
    if clf_name == 'LR':
        params = {
            'clf__penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
            'clf__C': trial.suggest_float('C', 0.05, 0.75),
            'clf__solver': trial.suggest_categorical('solver', ['saga']),  
            'clf__l1_ratio': trial.suggest_float('l1_ratio', 0, 1)
        }
    elif clf_name == 'SVM':
        params = {
             'clf__kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
             'clf__C': trial.suggest_float('C', 0.001, 0.05),
             'clf__degree': trial.suggest_int('degree', 2, 3),
             'clf__shrinking': trial.suggest_categorical('shrinking', [True, False]),
         }
    elif clf_name == 'RF':
        params = {
            'clf__max_depth': trial.suggest_int('max_depth', 2, 4, step=1),
            'clf__n_estimators': trial.suggest_int('n_estimators', 40, 400, step=20),
            'clf__criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        }
    elif clf_name == 'XGB':
        params = {
            'clf__learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            'clf__n_estimators': trial.suggest_int('n_estimators', 40, 400, step=20),
            'clf__booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'clf__gamma': trial.suggest_float('gamma', 7., 15.)
        }     
    
    return params

def objective(trial, scaling_methods, classifiers, X_train_outer, y_train_outer, cv_inner):
    '''
    Serves as the objective function for the Optuna hyperparameter optimization process. 
    It performs an inner cross-validation loop, fitting different combinations of scaling methods and classifiers, along with their respective hyperparameters, to the training data and evaluating the performance using ROC AUC score and log loss

    Parameters:
        trial (Optuna.trial.Trial): An Optuna trial object for hyperparameter optimization
        scaling_methods (list): List of scaling methods to be used
        classifiers (list): List of classifier names to be tested
        X_train_outer (ndarray): The outer training set features
        y_train_outer (ndarray): The outer training set labels
        cv_inner (sklearn.model_selection.StratifiedKFold): The StratifiedKFold cross-validator for the inner loop

    Returns:
        mean_val_roc_auc_inner (float): The mean ROC AUC score from the inner validation set
        emd_roc_auc (float): Earth Mover's Distance (EMD) between the train and validation ROC AUC scores for each trial.
    '''
    
    # Select scaler object available all_scalers function
    scaler_name = trial.suggest_categorical('scaler', scaling_methods)
    scalers_dict = all_scalers([scaler_name])
    scaler = scalers_dict[scaler_name]

    # Select classifier object from all_raw_clfs function
    classifier_name = trial.suggest_categorical('clf', classifiers)
    clfs_dict = all_raw_clfs([classifier_name])
    clf = clfs_dict[classifier_name]
    
    # Select classifier's hyperparameter from get_hyperparam_space function
    classifier_params = get_hyperparam_space(trial, classifier_name)
    
    # Create the pipeline
    pipeline = Pipeline(steps=[
    ('scaler', scaler),
    ('clf', clf)])
    
    pipeline.set_params(**classifier_params)
  
    # Perform inner cross-validation loop
    train_roc_auc_scores_inner = []
    val_roc_auc_scores_inner = []
    
    train_log_losses_inner = []
    val_log_losses_inner = []
    
    for train_index_inner, val_index_inner in cv_inner.split(X_train_outer, y_train_outer):
        X_train_inner, X_val_inner = X_train_outer[train_index_inner], X_train_outer[val_index_inner]
        y_train_inner, y_val_inner = y_train_outer[train_index_inner], y_train_outer[val_index_inner]
        
        # Fit the pipeline on the train set from the inner loop
        pipeline.fit(X_train_inner, y_train_inner)
           
        y_train_pred_inner = pipeline.predict_proba(X_train_inner)[:, 1]
        y_val_pred_inner = pipeline.predict_proba(X_val_inner)[:, 1]
        
        train_roc_auc_inner = roc_auc_score(y_train_inner, y_train_pred_inner)
        val_roc_auc_inner = roc_auc_score(y_val_inner, y_val_pred_inner)
        
        train_roc_auc_scores_inner.append(train_roc_auc_inner)
        val_roc_auc_scores_inner.append(val_roc_auc_inner)

        # Compute the log loss for the train and val set
        train_log_loss_inner = log_loss(y_train_inner, y_train_pred_inner)
        val_log_loss_inner = log_loss(y_val_inner, y_val_pred_inner)
        
        train_log_losses_inner.append(train_log_loss_inner)
        val_log_losses_inner.append(val_log_loss_inner)

    # Compute mean scores    
    mean_train_roc_auc_inner = np.mean(train_roc_auc_scores_inner)
    mean_val_roc_auc_inner = np.mean(val_roc_auc_scores_inner)
    mean_train_log_loss_inner = np.mean(train_log_losses_inner)
    mean_val_log_loss_inner = np.mean(val_log_losses_inner)
    
    # Compute the Earth Mover's Distance (EMD) for each trial
    emd_roc_auc = wasserstein_distance(train_roc_auc_scores_inner, val_roc_auc_scores_inner)
    
    # Set user attributes for training and validation scores
    trial.set_user_attr('train_roc_auc', train_roc_auc_scores_inner)
    trial.set_user_attr('val_roc_auc', val_roc_auc_scores_inner)
    trial.set_user_attr('mean_train_roc_auc', mean_train_roc_auc_inner)
    trial.set_user_attr('mean_val_roc_auc', mean_val_roc_auc_inner)
    trial.set_user_attr('mean_train_log_loss', mean_train_log_loss_inner)
    trial.set_user_attr('mean_val_log_loss', mean_val_log_loss_inner)
    trial.set_user_attr('EMD_roc_auc', emd_roc_auc)
    
    return mean_val_roc_auc_inner, emd_roc_auc

def tune_pipeline(scaling_methods, classifiers, X_train_outer, y_train_outer, n_trials, cv_inner):
    '''
    Utilizes Optuna to optimize a machine learning pipeline, which includes a scaling method, a classifier, and classifier hyperparameters. It returns the best-tuned pipeline based on Pareto optimality or the highest mean validation AUC-ROC score

    Parameters:
        scaling_methods (list): List of scaling methods to be used in the pipeline
        classifiers (list): List of classifier names to be used in the pipeline
        X_train_outer (np.ndarray): The outer training set features
        y_train_outer (np.ndarray): The outer training set labels
        n_trials (int): The number of trials for Optuna optimization
        cv_inner (sklearn.model_selection.StratifiedKFold): The StratifiedKFold cross-validator for the inner loop

    Returns:
        tuned_pipeline (sklearn.pipeline.Pipeline): The optimized pipeline with the best scaler and classifier
        best_params (dict): Dictionary of the best hyperparameters and scaler
        best_values (list): List of objective function values for the best trial
        df_pareto_trials (pd.DataFrame): DataFrame of the trials belonging to the Pareto front
    '''
    
    # Define the Optuna study and carry out multi-objective optimization
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0),
        directions=['maximize', 'minimize']) # [mean val AUC-ROC, mean val log loss]
    
    optuna.logging.set_verbosity(optuna.logging.WARN) # stop showing trial logging
    
    study.optimize(lambda trial: objective(trial, scaling_methods, classifiers, X_train_outer, y_train_outer, cv_inner),
                   n_trials=n_trials)
    
    # Get the Pareto front trials
    pareto_trials = study.best_trials

    # If pareto_trials is empty, get the trial with the highest mean validation AUC-ROC score
    if not pareto_trials:
        best_trial = max(study.trials, key=lambda trial: trial.values[0] if trial.values else float('-inf'))
    else:
        # Get the set of trial numbers on the Pareto front
        pareto_trial_numbers = set(trial.number for trial in pareto_trials)

        # Convert the study to a DataFrame
        df_trials = study.trials_dataframe()

        # Filter the DataFrame to include only the Pareto optimal trials
        df_pareto_trials = df_trials[df_trials['number'].isin(pareto_trial_numbers)]

        # Select the Pareto trial with the lowest mean validation log loss (i.e., proxy for best-calibrated pipeline)
        df_pareto_trials = df_pareto_trials.sort_values('user_attrs_mean_val_log_loss', ascending=True, ignore_index=True)
        best_trial_row = df_pareto_trials.loc[df_pareto_trials['user_attrs_mean_val_log_loss'].idxmin()]
        
        # Get the best trial from the Pareto front
        best_trial = study.trials[best_trial_row['number']]

    # Get the combination of best trial
    best_params = best_trial.params
    
    # Get the values of best trial
    best_values = best_trial.values
    
    # Retrieve the best scaler object
    best_scalers_dict = all_scalers([best_params['scaler']])
    best_scaler = best_scalers_dict[best_params['scaler']]
    
    # Retrieve the best classifier object
    best_clfs_dict = all_raw_clfs([best_params['clf']])
    best_clf = best_clfs_dict[best_params['clf']]

    # Retrieve the best hyperparameters of the classifier  (needs to be compatible with sklearn.pipeline)
    best_clf_params = {f"clf__{k}": v for k, v in best_params.items() if k not in ['scaler', 'clf']}
    
    # Create the pipeline with the best scaler and classifier objects and best classifier hyperparameters
    tuned_pipeline = Pipeline(steps=[
    ('scaler', best_scaler),
    ('clf', best_clf)])
    
    tuned_pipeline.set_params(**best_clf_params)
    
    return tuned_pipeline, best_params, best_values, df_pareto_trials

def nested_cv(scaling_methods, classifiers, feature_names, dataset, calibration, X_train, y_train, 
              n_splits_inner,  n_splits_outer, n_trials, print_results=True, 
              suppress_warnings=True, repeated_k_fold=False, 
              params=None):
    '''
    MASTER FUNCTION to tune, train and evaluate any given ML pipeline using nested cross-validation
    
    Paramaters:
        scaling_methods (list of str): List of desired scaling methods ('Min Max', 'Standard', 'Robust' or 'PQNlog')
        classifiers (list of str): List of desired classifiers ('LR', 'SVM', 'RF' or 'XGB')
        feature_names (list of str): Feature names of input X data
        dataset (str): Type of dataset ('Non-Augmented', 'Augmented')
        calibration (bool): False for no probability calibration and True for probability calibration using beta calibration
        X_train (pd.DataFrame): Unscaled input X data
        y_train (pd.DataFrame): Input y data
        n_splits_inner (int): Number of splits in the inner loop cross validation structure
        n_splits_outer (int): Number of splits in the outer loop cross validation structure
        n_trials (int): Number of trials (i.e., combinations of hyperparameters/iterations) to be used by Optuna
        print_results (bool): True to print AUC scores, False to not
        repeated_k_fold (bool): True to create a repeated inner loop cross validation structure
        params (dict or pd.DataFrame): optional hyperparamaters to feed to the model, such that no tuning is undertaken, only evaluation
    
    Returns:
        results (dict): All results including AUC distributions, mean, std, runtime, optimal hyperparameters for a given classifier, scaling method, cross validation combination
    '''
    
    # Suppress warnings:
    if suppress_warnings == True:
        warnings.filterwarnings('ignore')
        
    # Choose repeated or non-repeated cross-validation for inner loop:
    if repeated_k_fold == True:
        cv_inner = RepeatedStratifiedKFold(n_splits=n_splits_inner, n_repeats=3, random_state=SEED)
    elif repeated_k_fold == False:
        cv_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=SEED)
        
    # Select number of splits in cv_outer:
    cv_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=SEED)

    # Start timer:
    start = datetime.datetime.now()
    
    # Run nested-cross validation
    best_params_all = {}  # Initialize an empty dictionary to hold best parameters from each outer fold
    
    # Collect scores for uncalibrated and calibrated tuned pipeline
    train_roc_auc_scores_outer_uncalib = []
    test_roc_auc_scores_outer_uncalib = []
    train_roc_auc_scores_outer_calib = []
    test_roc_auc_scores_outer_calib = []
    
    roc_curve_calib = {}
    
    train_log_losses_outer_uncalib = []
    test_log_losses_outer_uncalib = []
    train_log_losses_outer_calib = []
    test_log_losses_outer_calib = []
    
    # Collect class labels and predictions to generate reliability diagram for uncalibrated and calibrated tuned pipeline
    y_true_all = []
    y_pred_all_uncalib = []
    y_pred_all_calib = []
    
    # If X_train and y_train are pandas DataFrame and Series respectively, convert them to numpy arrays
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    
    print('\x1B[4m' + 'Running all pipelines for Binary ' + dataset + ' dataset' + '\x1B[0m')
    # Outer cross-validation loop
    for fold_idx, (train_index_outer, val_index_outer) in enumerate(cv_outer.split(X_train, y_train)):
        X_train_outer, X_val_outer = X_train[train_index_outer], X_train[val_index_outer]
        y_train_outer, y_val_outer = y_train[train_index_outer], y_train[val_index_outer]
        
        # Stratified split of val_outer into calib_outer and test_outer
        X_calib_outer, X_test_outer, y_calib_outer, y_test_outer = train_test_split(
        X_val_outer, y_val_outer, test_size=0.4, stratify=y_val_outer, random_state=SEED)
        
        print(f'Outer Loop Fold {fold_idx+1}/{n_splits_outer} has started.')
        
        # Inner cross-validation loop to tune pipeline
        tuned_pipeline, best_params, best_values, df_pareto_trials = tune_pipeline(scaling_methods, classifiers, X_train_outer, y_train_outer, n_trials, cv_inner=cv_inner)
        
        # Save the best parameters for this fold
        best_params_all[fold_idx+1] = best_params
        
        # Get keys
        pipeline_keys = tuned_pipeline.named_steps.keys()

        # Combine the keys into a single string, separated by underscores
        pipeline_name = '_'.join(pipeline_keys)
        
        # Export Pareto trial information generated by all inner loop runs for a single outer loop iteration
        df_pareto_trials.to_excel(dataset + '_' + pipeline_name + '_df_pareto_trials_Outer_Fold' + str(fold_idx+1) + ' - Binary.xlsx')
        
        # Fit the tuned pipeline on the train set from the outer loop
        tuned_pipeline.fit(X_train_outer, y_train_outer)
        
        if calibration:
            calib_tuned_pipeline = CalibratedPipeline(base_pipeline=tuned_pipeline)
            
            # Fit calibrated tuned pipeline
            calib_tuned_pipeline.fit(X_calib_outer, y_calib_outer, refit_estimator=False)
        else:
            # If no probability calibration, then continue using the fitted tuned_pipeline for predictions
            calib_tuned_pipeline = deepcopy(tuned_pipeline)
            
        # Make predictions on the train_outer and test_outer set using uncalibrated and calibrated tuned pipeline
        y_train_pred_outer_uncalib = tuned_pipeline.predict_proba(X_train_outer)[:, 1]
        y_test_pred_outer_uncalib = tuned_pipeline.predict_proba(X_test_outer)[:, 1]
        y_train_pred_outer_calib = calib_tuned_pipeline.predict_proba(X_train_outer)[:, 1]
        y_test_pred_outer_calib = calib_tuned_pipeline.predict_proba(X_test_outer)[:, 1]

        # Compute the roc_auc for the train_outer and test_outer set using uncalibrated and calibrated tuned pipeline
        train_roc_auc_outer_uncalib = roc_auc_score(y_train_outer, y_train_pred_outer_uncalib)
        test_roc_auc_outer_uncalib = roc_auc_score(y_test_outer, y_test_pred_outer_uncalib)
        train_roc_auc_outer_calib = roc_auc_score(y_train_outer, y_train_pred_outer_calib)
        test_roc_auc_outer_calib = roc_auc_score(y_test_outer, y_test_pred_outer_calib)
        
        # Compute FPR, TPR, thresholds for only the test outer set using using only the calibrated tuned pipeline
        fpr_calib, tpr_calib, thresholds_calib = roc_curve(y_test_outer, y_test_pred_outer_calib)
        roc_curve_calib['Outer Loop Fold ' + str(fold_idx+1)] = [fpr_calib, tpr_calib, test_roc_auc_outer_calib]   

        
        # Collect roc_auc scores for train_outer and test_outer set using uncalibrated and calibrated tuned pipeline
        train_roc_auc_scores_outer_uncalib.append(train_roc_auc_outer_uncalib)
        test_roc_auc_scores_outer_uncalib.append(test_roc_auc_outer_uncalib)
        train_roc_auc_scores_outer_calib.append(train_roc_auc_outer_calib)
        test_roc_auc_scores_outer_calib.append(test_roc_auc_outer_calib)
        
        
        # Compute the log loss for train_outer and test_outer set using uncalibrated and calibrated tuned pipeline
        train_log_loss_outer_uncalib = log_loss(y_train_outer, y_train_pred_outer_uncalib)
        test_log_loss_outer_uncalib = log_loss(y_test_outer, y_test_pred_outer_uncalib)
        train_log_loss_outer_calib = log_loss(y_train_outer, y_train_pred_outer_calib)
        test_log_loss_outer_calib = log_loss(y_test_outer, y_test_pred_outer_calib)
        
        # Collect log_losses for train_outer and test_outer set using uncalibrated and calibrated tuned pipeline
        train_log_losses_outer_uncalib.append(train_log_loss_outer_uncalib)
        test_log_losses_outer_uncalib.append(test_log_loss_outer_uncalib)
        train_log_losses_outer_calib.append(train_log_loss_outer_calib)
        test_log_losses_outer_calib.append(test_log_loss_outer_calib)
        
        # Store the true labels and predicted probabilities for uncalibrated and calibrated tuned pipeline
        y_true_all.extend(y_test_outer)
        y_pred_all_uncalib.extend(y_test_pred_outer_uncalib)
        y_pred_all_calib.extend(y_test_pred_outer_calib)
        
    
    # Convert the dictionary of best parameters to a pandas DataFrame
    df_best_params = pd.DataFrame.from_dict(best_params_all, orient='index')

    # Export the DataFrame to an Excel file
    df_best_params.to_excel(dataset + ' - Best_Pipelines - Binary.xlsx')
    
    # Compute mean scores for calibrated tuned pipeline
    mean_train_roc_auc_outer_calib = np.mean(train_roc_auc_scores_outer_calib)
    mean_test_roc_auc_outer_calib = np.mean(test_roc_auc_scores_outer_calib)
    
    mean_train_log_loss_outer_calib = np.mean(train_log_losses_outer_calib)
    mean_test_log_loss_outer_calib = np.mean(test_log_losses_outer_calib)
    
    # Compute standard deviation of scores for calibrated tuned pipeline   
    std_train_roc_auc_outer_calib = np.std(train_roc_auc_scores_outer_calib)
    std_test_roc_auc_outer_calib = np.std(test_roc_auc_scores_outer_calib)
    
    std_train_log_loss_outer_calib = np.std(train_log_losses_outer_calib)
    std_test_log_loss_outer_calib = np.std(test_log_losses_outer_calib)
    
    # Compute the 95% confidence interval for the training and test AUC-ROC score using calibrated tuned pipeline
    confidence_level = 0.95
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    
    margin_of_error_train = z * (std_train_roc_auc_outer_calib / np.sqrt(n_splits_outer))
    confidence_interval_train = (mean_train_roc_auc_outer_calib - margin_of_error_train, mean_train_roc_auc_outer_calib + margin_of_error_train)

    margin_of_error_test = z * (std_test_roc_auc_outer_calib / np.sqrt(n_splits_outer))
    confidence_interval_test = (mean_test_roc_auc_outer_calib - margin_of_error_test, mean_test_roc_auc_outer_calib + margin_of_error_test)
    
    # End timer and calculate runtime:
    end = datetime.datetime.now()
    runtime = end - start

    # Display results:
    if print_results==True:
        print('Time elapsed: ' + str(runtime))
        print('95% Confidence Interval for Training AUC-ROC Score =', (round(confidence_interval_train[0], 3), round(confidence_interval_train[1], 3)))
        print('95% Confidence Interval for Test AUC-ROC Score =', (round(confidence_interval_test[0], 3), round(confidence_interval_test[1], 3)))

    # Return results in a dictionary:
    eval_results = {
        'Runtime': runtime,
        'Mean Test AUC-ROC': mean_test_roc_auc_outer_calib,
        'SD Test AUC-ROC': std_test_roc_auc_outer_calib,
        'Mean Training AUC-ROC': mean_train_roc_auc_outer_calib,
        'SD Training AUC-ROC': std_train_roc_auc_outer_calib,
        'Mean Test log loss': mean_test_log_loss_outer_calib,
        'SD Test log loss': std_test_log_loss_outer_calib,
        'Mean Training log loss': mean_train_log_loss_outer_calib,
        'SD Training log loss': std_train_log_loss_outer_calib
    }
    
    # Store performance evaluation results and plot roc curve
    eval_results_df = pd.DataFrame.from_dict(eval_results, orient='index')
    eval_results_df.to_excel(dataset + ' - Performance Evaluation -' + ' Binary.xlsx')
    
    mean_roc_curve_binary_fig = plot_roc_auc_curve_binary(roc_curve_calib, dataset)
    mean_roc_curve_binary_fig.savefig(dataset + ' - Mean ROC Curve' + ' Binary.png',
                            format='png',
                            bbox_inches='tight',
                            orientation='landscape',
                            dpi=1200)
    
    if calibration:
        # Compute expected calibration error (ECE) and fraction of positives and mean predicted probabilities for uncalibrated and calibrated tuned pipeline
        ECE_score_uncalib = ECE(y_true_all, y_pred_all_uncalib, n_bins=10)
        log_loss_uncalib = log_loss(y_true_all, y_pred_all_uncalib)
        fraction_of_positives_uncalib, mean_predicted_value_uncalib = calibration_curve(y_true_all, y_pred_all_uncalib, n_bins=10, strategy='quantile')
        calibration_info_uncalib = (fraction_of_positives_uncalib, mean_predicted_value_uncalib, ECE_score_uncalib, log_loss_uncalib)
        
        ECE_score_calib = ECE(y_true_all, y_pred_all_calib, n_bins=10)
        log_loss_calib = log_loss(y_true_all, y_pred_all_calib)
        fraction_of_positives_calib, mean_predicted_value_calib = calibration_curve(y_true_all, y_pred_all_calib, n_bins=10, strategy='quantile')
        calibration_info_calib = (fraction_of_positives_calib, mean_predicted_value_calib, ECE_score_calib, log_loss_calib)
    
        # Plot and save calibration curves
        calibration_fig = plot_reliability_diagram(calibration,
                                                   calibration_info_uncalib,
                                                   calibration_info_calib, 
                                                   dataset)
        
        calibration_fig.savefig(dataset + ' - Reliability Diagram' + '.png',
                                format='png',
                                bbox_inches='tight',
                                orientation='landscape',
                                dpi=1200)
    
    else:
        calibration_info_uncalib = None
        calibration_info_calib = None
    
    return (eval_results, calibration_info_uncalib, calibration_info_calib)

# Methods to evaluate:
scaling_methods = ['Min Max', 'Standard', 'None']
classifiers = ['LR', 'SVM', 'RF', 'XGB']

with open('EVALUATION HAS STARTED.txt', 'w') as fp:
    pass

# Non-augmented binary run:
results_non_augmented_binary = nested_cv(scaling_methods=scaling_methods,
                                          classifiers=classifiers,
                                          feature_names=feature_names,
                                          dataset='Non-Augmented',
                                          calibration=True,
                                          X_train=X_train_non_augmented,
                                          y_train=y_train_non_augmented,
                                          n_splits_inner=5,
                                          n_splits_outer=5,
                                          n_trials=100)

df_non_aug_eval, d_non_aug_calib_info_uncalib, d_non_aug_calib_info_calib = results_non_augmented_binary

# Produce final non-augmented binary tuned pipeline
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Split the data into 90% training and 10% calibration sets
X_train_non_aug, X_calib_non_aug, y_train_non_aug, y_calib_non_aug = train_test_split(X_train_non_augmented,
                                                                                      y_train_non_augmented,
                                                                                      test_size=0.10, 
                                                                                      random_state=SEED)

X_train_outer = X_train_non_aug.to_numpy()
y_train_outer = y_train_non_aug.to_numpy()
n_trials=100

print('Training for Calibrated Tuned Pipeline using the Non-Augmented Binary Dataset has started.')
start = datetime.datetime.now()
tuned_pipeline_non_aug_binary, best_params_non_aug_binary, best_values_non_aug_binary, df_pareto_trials_non_aug_binary = tune_pipeline(scaling_methods, classifiers, X_train_outer, y_train_outer, n_trials, cv_inner=cv_inner)
tuned_pipeline_non_aug_binary.fit(X_train_outer, y_train_outer)
calib_tuned_pipeline_non_aug_binary = CalibratedPipeline(base_pipeline=tuned_pipeline_non_aug_binary)
calib_tuned_pipeline_non_aug_binary.fit(X_calib_non_aug, y_calib_non_aug)
    
dump(calib_tuned_pipeline_non_aug_binary, 'calib_tuned_pipeline_non_aug_binary.joblib')
df_pareto_trials_non_aug_binary.to_excel('Non-Augmented - Calibrated Tuned Pipeline - Binary - Pareto - Final.xlsx')
end = datetime.datetime.now()
runtime = end - start
print('Training for Calibrated Tuned Pipeline using the Non-Augmented Binary Dataset has finished.')
print('Time elapsed: ' + str(runtime))

# Augmented binary run:
results_augmented_binary = nested_cv(scaling_methods=scaling_methods,
                                      classifiers=classifiers,
                                      feature_names=feature_names,
                                      dataset='Augmented',
                                      calibration=True,
                                      X_train=X_train_augmented,
                                      y_train=y_train_augmented,
                                      n_splits_inner=5,
                                      n_splits_outer=5,
                                      n_trials=100)   

df_aug_eval, d_aug_calib_info_uncalib, d_aug_calib_info_calib = results_augmented_binary

# Produce final non-augmented binary tuned pipeline
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Split the data into 90% training and 10% calibration sets
X_train_aug, X_calib_aug, y_train_aug, y_calib_aug = train_test_split(X_train_augmented,
                                                                      y_train_augmented,
                                                                      test_size=0.10, 
                                                                      random_state=SEED)

X_train_outer_aug = X_train_aug.to_numpy()
y_train_outer_aug = y_train_aug.to_numpy()

n_trials=100

print('Training for Calibrated Tuned Pipeline using the Augmented Binary Dataset has started.')
start = datetime.datetime.now()
tuned_pipeline_aug_binary, best_params_aug_binary, best_values_aug_binary, df_pareto_trials_aug_binary = tune_pipeline(scaling_methods, classifiers, X_train_outer_aug, y_train_outer_aug, n_trials, cv_inner=cv_inner)
tuned_pipeline_aug_binary.fit(X_train_outer_aug, y_train_outer_aug)
calib_tuned_pipeline_aug_binary = CalibratedPipeline(base_pipeline=tuned_pipeline_aug_binary)
calib_tuned_pipeline_aug_binary.fit(X_calib_aug, y_calib_aug)

    
dump(calib_tuned_pipeline_aug_binary, 'calib_tuned_pipeline_aug_binary.joblib')
df_pareto_trials_aug_binary.to_excel('Augmented - Calibrated Tuned Pipeline - Binary - Pareto - Final.xlsx')
end = datetime.datetime.now()
runtime = end - start
print('Training for Calibrated Tuned Pipeline using the Augmented Binary Dataset has finished.')
print('Time elapsed: ' + str(runtime))

'''
ALE for Augmented Binary Dataset
'''
pipeline = load('calib_tuned_pipeline_aug_binary.joblib')

if pipeline.is_fitted:
    print("The pipeline is fitted and calibrated.")
else:
    print("The pipeline is not fitted or calibrated.")

# define prediction function wrapping the fitted and calibrated pipeline
predict_fn = lambda x: pipeline.predict_proba(x)[:, 1]

# initialize and fit the explainer on your training data (90% of augmented dataset)
GP_names = X_train_aug.columns.tolist()
ale = ALE(predict_fn, feature_names=GP_names)
exp = ale.explain(X_train_aug.values)

# Initialize an empty dictionary to store the feature importance
feature_importance = {}

# Calculate the importance for each feature
for i, feature in enumerate(GP_names):
    ale_values = exp.ale_values[i]
    importance = np.max(ale_values) - np.min(ale_values)
    feature_importance[feature] = importance

# Sort features by importance
feature_importance_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance_sorted:
    print(f"{feature}: {importance}")
     
fig, ax = plt.subplots()

# ALE plot created for GP8, similarly for GP9
plot_ale(exp, features=['GP8'], ax=ax, line_kw={'label': 'Probability of "CRC" class'})

# Set y-axis label
ax.set_ylabel('Effect on calibrated predicted probability (centered)')
ax.set_xlabel('Log-transformed GP8 (FA2[6]G1)')
ax.legend(loc='best')

# Show grid
ax.grid(False)

df_train_smote_augmented_exp = df_train_smote_augmented.filter(like='GP').apply(np.exp)

# Save the figure
fig.savefig('ALE_plot - GP8.png',
            format='png',
            bbox_inches='tight',
            dpi=1200)