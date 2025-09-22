# optuna_optimizer.py
import os
import argparse
import logging
import torch
import torch.optim as optim
import optuna
from data_loader import load_and_preprocess_data
from pinn_model import PINN_Model
from trainer import train_model
from utils import seed_everything
from pbm_module import PROCESS_BASED_MODEL

#  Logger Setup 
LOGGER_NAME = "Optuna_Logger"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d – %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

#  Global variables to hold data, to avoid reloading on each trial 
train_loader = None
val_loader = None
input_dim = None
y_scaler = None
pbm_simulator = None
device = None

def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the optimization script."""
    parser = argparse.ArgumentParser(description="Optimize PINN hyperparameters using Optuna.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file (Excel).')
    parser.add_argument('--output_dir', type=str, default='optuna_runs', help='Directory to save optimization results.')
    parser.add_argument('--config_path', type=str, default='PBM_config.yaml', help='Path to the PBM config file.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of optimization trials to run.')
    return parser.parse_args()

def objective(trial: optuna.trial.Trial) -> float:
    """Defines a single hyperparameter optimization trial for Optuna."""
    #  Hyperparameter Search Space 
    # PBM parameters to optimize
    diff_params = trial.suggest_float('diff_params', 1e-10, 1e-8, log=True)
    k0_n2o = trial.suggest_float('k0_n2o', 0.01, 0.1, log=True)
    init_on = trial.suggest_float('init_on', 0.5, 2.0)
    ph = trial.suggest_float('ph', 6.5, 7.5)

    pbm_params = {
        'diff_params': diff_params, 'k0_n2o': k0_n2o, 'init_on': init_on, 'ph': ph
    }

    # PINN (LSTM) parameters
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
    lstm_hidden_dim = trial.suggest_categorical('lstm_hidden_dim', [64, 128, 256])
    fc_hidden_dim = trial.suggest_categorical('fc_hidden_dim', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    lambda_pbm = trial.suggest_float('lambda_pbm', 0.1, 1.0)
    
    #  Model Initialization 
    model = PINN_Model(
        input_dim=input_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        fc_hidden_dim=fc_hidden_dim,
        output_dim=1,
        num_lstm_layers=num_lstm_layers
    ).to(device)

    #  Training for the trial 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create a temporary directory for this trial's outputs
    os.makedirs(args.output_dir, exist_ok=True)
    _, history = train_model(
        model=model,
        pbm_simulator=pbm_simulator,
        pbm_params=pbm_params,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=50,
        lambda_pbm=lambda_pbm,
        patience=10,    
        device=device,
        output_dir=args.output_dir,
        y_scaler=y_scaler
    )

    # Return the best validation loss for Optuna to minimize
    return min(history['val_loss'])

if __name__ == "__main__":
    args = _parse_args()
    
    #  Global Setup (Load data once to speed up trials) 
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Populate global variables with loaded data
    train_loader, val_loader, _, input_dim, y_scaler = load_and_preprocess_data(
        file_path=args.data_path,
        seq_length_hours=240
    )
    pbm_simulator = PROCESS_BASED_MODEL(config_path=args.config_path)

    #  Run Optuna Study 
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("\n" + "="*50)
    logger.info("Best trial found:")
    trial = study.best_trial
    logger.info(f"  Value (Min Validation Loss): {trial.value:.6f}")
    logger.info("  Optimal Parameters: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
