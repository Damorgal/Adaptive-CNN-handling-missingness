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

def generate_ratio_features(df, duration=48.0, timestep=1.0):
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

        # Extracting measurements
        feature_dict = {}
        for variable in timeseries_variables:
            measurements = df_t[df_t['Variable'] == variable].Value
            
            # Times that measurements were done ----------------------------
            aux = np.array([1 for i in measurements if not np.isnan(i)])
            feature_dict['times_' + variable] = float(int(len(aux)))
            
            
            # Distance between measures ------------------------------------
            #mean = np.mean(measurements)
            #times = df_t[df_t['Variable'] == variable].Time
            #if np.isnan(mean):
            #    feature_dict['distance_' + variable] = t_end - last_time[variable]
            #else:
            #    last_time[variable] = times.iloc[-1]
            #    #feature_dict['distance_' + variable] = times.iloc[0] - last_time[variable]
            #    feature_dict['distance_' + variable] = t_end - last_time[variable]
            
            # Mean of measurements -----------------------------------------
            #feature_dict['mean_' + variable] = np.mean(measurements)
            #feature_dict['missing_' + variable] = int(len(measurements) == 0)

        features_list.append(feature_dict)
        ## Solution
    
    # Create a table with (row - timesteps) and (column - features)
    df_out = pd.DataFrame(features_list)
    
    # Intra-patient imputation by forward-filling 
    df_out = df_out.ffill()
    return df_out

if __name__ == '__main__':
    data_path = 'data/'
    N = 10000
    df_labels = pd.read_csv(data_path + 'labels.csv')
    df_labels = df_labels[:N]
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv(data_path + 'files/{}.csv'.format(i))
        
    # This is for the ratio between measurements in each interval and the average of
    # measurements in each interval ------------------------------------------------
    
    features = Parallel(n_jobs=16)(delayed(generate_ratio_features)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    pickle.dump([features, df_labels], open(data_path + 'features_labels_ratio-meas-avg_{}.p'.format(N), 'wb'))
    
    with open(data_path + 'features_labels_ratio-meas-avg_{}.p'.format(N), 'rb') as f:
        features, df_labels = pickle.load(f)
        X = np.array([df_i.values for df_i in features])
        y = df_labels['In-hospital_death'].values.copy()
        y[y == -1] = 0

    print(X.shape, y.shape)
    
    for ind in range(48):
        avg = np.mean(X[:,ind,:])
        X[:,ind,:] = X[:,ind,:]/avg
    
    np.savez(open(data_path + 'data_ratio.npz', 'wb'), X=X, y=y)
   
