import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import Parallel, delayed
from random import randint

import yaml
config = yaml.load(open('lib/config.yaml'))
static_variables = config['invariant']
timeseries_variables = config['timeseries']

def generate_features(df, duration=48.0, timestep=1.0):
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a multivariate time series.
    
    Args:
        df: pd.DataFrame, with columns [Time, Variable, Value]
    
    Returns:
        df_out: pd.DataFrame. see sample output.
    """
    # Remove unknow values
    df = df[df.Value >= 0].replace({-1: np.nan}).dropna()

    # Convert time to fractional hours
    df['Time'] = df['Time'].apply(lambda s: int(s.split(':')[0]) + int(s.split(':')[1])/60)

    # Consider static variables
    df = df.iloc[:5]

    features_list = []
    
    # Loop through all time windows
    for t in np.arange(0.0, duration, timestep):
        
        feature_dict = {}
        for variable in static_variables:
            feature_dict['static' + variable] = aux = np.mean(df[df.Variable == variable].Value)
            feature_dict['missing_' + variable] = int(np.isnan(aux)) # Mean is not necesary beacause is just one #
                                                                     # but is necesary to convert the format of the value 
        features_list.append(feature_dict)
        ## Solution
    
    # Create a table with (row - timesteps) and (column - features)
    df_out = pd.DataFrame(features_list)
    
    # Intra-patient imputation by forward-filling 
    df_out = df_out.ffill()
    return df_out

def impute_missing_values(X, by='mean'):
    """
    For each feature column, impute missing values  (np.nan) with the 
    population mean for that feature.
    
    Args:
        X: np.array, shape (N, L, d). X could contain missing values
    Returns:
        X: np.array, shape (N, L, d). X does not contain any missing values
    """
    N, L, d = X.shape
    X = X.reshape(N*L, d)
    if by == 'mean':
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        
    elif by == 'CF':
        for j in range(d):
            last = 0   # Last value imputed
            for i in range(N*L):
                if np.isnan(X[i,j]):
                    if i == 0: # First case false
                        for k in range(N*L):
                            if not np.isnan(X[k,j]):
                                last = X[k,j]
                                break
                    # Imputation
                    X[i,j] = last
                last = X[i,j]
                
    if by == 'mw-mean': #[0,1,2,3,4] = [Age,Gender,Height,ICUType,Weight]
        miss_gender = []
        men = []
        women = []
        for i in range(X.shape[0]):
            if X[i,1] == 1:
                men.append(True)
                women.append(False)
            elif X[i,1] == 0:
                women.append(True)
                men.append(False)
            else:
                men.append(False)
                women.append(False)
                miss_gender.append(i)
        men = np.array(men)
        women = np.array(women)
        miss_gender = np.array(miss_gender)
        m_col_means = np.nanmean(X[men], axis=0)
        w_col_means = np.nanmean(X[women], axis=0)
        # Impute gender based on height and weight, else random
        for m in miss_gender:
            # Imputation by height
            if not np.isnan(X[m,2]):
                if np.abs(X[m,2]-m_col_means[2]) < np.abs(X[m,2]-w_col_means[2]):
                    X[m,1] = 1
                else:
                    X[m,1] = 0
            # Imputation by weight
            elif not np.isnan(X[m,4]):
                if np.abs(X[m,4]-m_col_means[4]) < np.abs(X[m,4]-w_col_means[4]):
                    X[m,1] = 1
                else:
                    X[m,1] = 0
            # Random imputation
            else:
                X[m,1] = randint(0,1)
         
        men = []
        women = []
        for i in range(X.shape[0]):
            if X[i,1] == 1.:
                men.append(i)
            elif X[i,1] == 0.:
                women.append(i)
        men = np.array(men)
        women = np.array(women)
        # Impute missing invariant features based on gender
        for m in men:
            if np.isnan(X[m,2]):
                X[m,2] = m_col_means[2]
            if np.isnan(X[m,4]):
                X[m,4] = m_col_means[4] 
        for m in women:
            if np.isnan(X[m,2]):
                X[m,2] = w_col_means[2]
            if np.isnan(X[m,4]):
                X[m,4] = w_col_means[4]
        print(np.where(np.isnan(X)))
                        
    return X.reshape(N, L, d)

def standardize_features(X):
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: np.array, shape (N, d).
    Returns:
        X: np.array, shape (N, d). Values are normalized per column.
    """
    scaler = StandardScaler()
    N, L, d = X.shape
    X = X.reshape(N*L, d)
    X = scaler.fit_transform(X)
    return X.reshape(N, L, d)

if __name__ == '__main__':
    data_path = 'data/'
    N = 10000
    df_labels = pd.read_csv(data_path + 'labels.csv')
    df_labels = df_labels[:N]
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv(data_path + 'files/{}.csv'.format(i))
    
    features = Parallel(n_jobs=16)(delayed(generate_features)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    pickle.dump([features, df_labels], open(data_path + 'features_labels_inv_{}.p'.format(N), 'wb'))
    
    with open(data_path + 'features_labels_inv_{}.p'.format(N), 'rb') as f:
        features, df_labels = pickle.load(f)
        X = np.array([df_i.values for df_i in features])
        y = df_labels['In-hospital_death'].values.copy()
        y[y == -1] = 0
    
    print(X.shape, y.shape)
    
    X_miss = X[:,:,:5]
    X_nmiss = X[:,:,5:]
    
    X_nmiss = impute_missing_values(X_nmiss,'mw-mean')
    X_nmiss = standardize_features(X_nmiss)
    
    np.savez(open(data_path + 'data_time-inv.npz', 'wb'), X=X_nmiss, y=y)
    np.savez(open(data_path + 'data_time-inv_miss.npz', 'wb'), X=X_miss, y=y)
