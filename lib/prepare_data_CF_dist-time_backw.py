import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import Parallel, delayed

import yaml
config = yaml.load(open('lib/config.yaml'))
static_variables = config['invariant']
timeseries_variables = config['timeseries']

def generate_time_features(df, duration=48.0, timestep=1.0):
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

    # Consider only time series (time-varying variables)
    df = df.iloc[5:]

    features_list = []
    
    # Loop through all time windows
    last_time = {}
    for var in timeseries_variables:
        last_time[var] = 0.0
    for t in np.arange(0.0, duration, timestep):
        # Extract data within this time window
        t_start, t_end = t, t+timestep
        df_t = df[(t_start < df['Time']) & (df['Time'] <= t_end)]

        # TODO: generate a feature dictionary (similar to Project 1) 
        # for this time window
        # e.g. {
        #     'mean_HR': ??, 'missing_HR': ??, 
        #     'mean_RR': ??, 'missing_RR': ??,
        #     ...
        # }
        
        ################### Just select two of the tree options below ########################
        feature_dict = {}
        for variable in timeseries_variables:
            measurements = df_t[df_t['Variable'] == variable].Value
            
            # Distance between measures ------------------------------------
            mean = np.mean(measurements)
            times = df_t[df_t['Variable'] == variable].Time
            if np.isnan(mean):
                feature_dict['distance_' + variable] = t_end - last_time[variable]
            else:
                last_time[variable] = times.iloc[-1]
                #feature_dict['distance_' + variable] = times.iloc[0] - last_time[variable]
                feature_dict['distance_' + variable] = t_end - last_time[variable]
            
            # Mean of measurements -----------------------------------------
            #feature_dict['mean_' + variable] = np.mean(measurements)
            #feature_dict['missing_' + variable] = int(len(measurements) == 0)
            
            # Times that measurements were done ----------------------------
            aux = np.array([1 for i in measurements if not np.isnan(i)])
            #feature_dict['times_' + variable] = int(len(aux))          
            feature_dict['times_' + variable] = int(len(aux))-int(len(measurements)-len(aux))

        features_list.append(feature_dict)
        ## Solution
    
    # Create a table with (row - timesteps) and (column - features)
    df_out = pd.DataFrame(features_list)
    
    # Intra-patient imputation by forward-filling 
    df_out = df_out.ffill()
    return df_out

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

    # Consider only time series (time-varying variables)
    df = df.iloc[5:]

    features_list = []
    
    # Loop through all time windows
    for t in np.arange(0.0, duration, timestep):
        # Extract data within this time window
        t_start, t_end = t, t+timestep
        df_t = df[(t_start < df['Time']) & (df['Time'] < t_end)]

        # TODO: generate a feature dictionary (similar to Project 1) 
        # for this time window
        # e.g. {
        #     'mean_HR': ??, 'missing_HR': ??, 
        #     'mean_RR': ??, 'missing_RR': ??,
        #     ...
        # }
        
        ## Solution
        feature_dict = {}
        for variable in timeseries_variables:
            measurements = df_t[df_t['Variable'] == variable].Value
            feature_dict['mean_' + variable] = np.mean(measurements)
            feature_dict['missing_' + variable] = int(len(measurements) == 0)

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
        
    ## UNCOMENT PROGRAM THAT YOU DESIRE #####################################################
        
    # This is for the features ----------------------------------------------
    #features = Parallel(n_jobs=16)(delayed(generate_features)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    #pickle.dump([features, df_labels], open(data_path + 'features_labels_missing_{}.p'.format(N), 'wb'))
    
    #with open(data_path + 'features_labels_missing_{}.p'.format(N), 'rb') as f:
     #   features, df_labels = pickle.load(f)
      #  X = np.array([df_i.values for df_i in features])
       # y = df_labels['In-hospital_death'].values.copy()
        #y[y == -1] = 0
    #print(X.shape, y.shape)
    
    #X_miss = X[:,:,35:]
    #X_nmiss = X[:,:,:35]
    #X_nmiss = impute_missing_values(X_nmiss,'CF')
    #X_nmiss = standardize_features(X_nmiss)
    
    #np.savez(open(data_path + 'data_nmiss_CF.npz', 'wb'), X=X_nmiss, y=y)
    #np.savez(open(data_path + 'data_miss_CF.npz', 'wb'), X=X_miss, y=y)

    # This is for the times and distance --------------------------------------
    features = Parallel(n_jobs=16)(delayed(generate_time_features)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    pickle.dump([features, df_labels], open(data_path + 'features_labels_dist_times_backw_{}.p'.format(N), 'wb'))
    
    with open(data_path + 'features_labels_dist_times_backw_{}.p'.format(N), 'rb') as f:
        features, df_labels = pickle.load(f)
        X = np.array([df_i.values for df_i in features])
        y = df_labels['In-hospital_death'].values.copy()
        y[y == -1] = 0

    print(X.shape, y.shape)
    
    X_dist = X[:,:,:35]
    X_times = X[:,:,35:]
    #X_nmiss = impute_missing_values(X_nmiss,'CF')
    #X_nmiss = standardize_features(X_nmiss)
    
    np.savez(open(data_path + 'data_dist_backw.npz', 'wb'), X=X_dist, y=y)
    np.savez(open(data_path + 'data_times_backw_neg.npz', 'wb'), X=X_times, y=y)
   
