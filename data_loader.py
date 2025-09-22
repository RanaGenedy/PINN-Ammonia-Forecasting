# data_loader.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

def _create_data_sequences(features, target, seq_length_hours = 240):
    """
    Transforms the features and target into sequences for LSTM input.
    """
    X_sequences, y_sequences = [], []
    for i in range(seq_length_hours, len(features)):
        # Grab the past `seq_length_hours` hours of features
        X_sequences.append(features[i-seq_length_hours:i])
        # The target is the value at the current time step
        y_sequences.append(target[i])
        
    
    return np.array(X_sequences), np.array(y_sequences)
    
def load_and_preprocess_data(file_path, batch_size=64, seq_length_hours=240):
    """Loads, preprocesses, and splits the data into loaders."""
    df = pd.read_csv(file_path)
    
    df['D/T'] = pd.to_datetime(df['D/T'])
    df = df[['D/T', 'Month', 'AAT', 'WS', 'WD', 'RH', 'SR', 'Agitation', 'NH3_mg/m2/d']]
    df.set_index('D/T', inplace=True)
    
    # Resampling to hourly data and handling missing values
    df = df.resample('h').mean().interpolate(method='linear').dropna()
    # Ensure dataset is large enough, if not, raise an error
    if len(df) < seq_length_hours + 10: # Add a small buffer
        raise ValueError(
            f"Dataset is too small ({len(df)} rows) to create sequences with a lookback of {seq_length_hours} hours. "
            f"Please use a larger dataset or reduce the --seq_length_hours argument."
        )
    # Separating features and target
    features_df = df[['Month', 'AAT', 'WS', 'WD', 'RH', 'SR', 'Agitation']].values
    target_df = df[['NH3_mg/m2/d']].values

    # Train, validation, test split
    train_len = int(0.7 * len(df))
    val_len = int(0.15 * len(df))
    test_len = len(df) - train_len - val_len
    
    X_train_raw = features_df[:train_len]
    y_train_raw = target_df[:train_len]
    
    X_val_raw = features_df[train_len:train_len+val_len]
    y_val_raw = target_df[train_len:train_len+val_len]
    
    X_test_raw = features_df[train_len+val_len:]
    y_test_raw = target_df[train_len+val_len:]
    
    # Features scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train_raw) # Only fit on training data
    X_val_scaled = sc.transform(X_val_raw)
    X_test_scaled = sc.transform(X_test_raw)
    
    # Target scaling 
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    y_val_scaled = y_scaler.transform(y_val_raw)
    y_test_scaled = y_scaler.transform(y_test_raw)

    # Create Sequences 
    X_train_seq, y_train_seq = _create_data_sequences(X_train_scaled, y_train_scaled, seq_length_hours)
    X_val_seq, y_val_seq = _create_data_sequences(X_val_scaled, y_val_scaled, seq_length_hours)
    X_test_seq, y_test_seq = _create_data_sequences(X_test_scaled, y_test_scaled, seq_length_hours)    

    # Create DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train_seq.shape[2], y_scaler