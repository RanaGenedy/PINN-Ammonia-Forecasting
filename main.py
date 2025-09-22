# main.py
import argparse
import logging
import os
from datetime import datetime
import torch
import torch.optim as optim
from data_loader import load_and_preprocess_data
from pinn_model import PINN_Model
from trainer import train_model, evaluate_model
from utils import seed_everything
from pbm_module import PROCESS_BASED_MODEL

# Logger Setup
LOGGER_NAME = "PINN_Logger"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d – %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Physics-Informed Neural Network (PINN) with an LSTM.")
    
    # Data and Config Paths
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file (CSV format).')
    parser.add_argument('--config_path', type=str, default='PBM_config.yaml', help='Path to the PBM configuration file.')
    parser.add_argument('--output_dir', type=str, default='runs', help='Directory to save plots and results.')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--lambda_pbm', type=float, default=0.2, help='Weighting factor for the PBM component of the loss.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping.')
    
    # Model Architecture Hyperparameters
    parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='Number of hidden units in the LSTM layer.')
    parser.add_argument('--fc_hidden_dim', type=int, default=64, help='Number of hidden units in the fully connected layer.')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of layers in the LSTM.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate.')

    return parser.parse_args()

def main():
    # Setup and Argument Parsing
    args = _parse_args()
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a unique, timestamped directory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Output will be saved to: {run_dir}")
    logger.info(f"Running with arguments: {args}")

    # Data Loading
    train_loader, val_loader, test_loader, input_dim, y_scaler = load_and_preprocess_data(
        file_path=args.data_path,
        batch_size=args.batch_size,
        seq_length_hours=168,  # 7 days of hourly data
    )

    # Model and PBM Initialization
    pinn = PINN_Model(
        input_dim=input_dim, 
        lstm_hidden_dim=args.lstm_hidden_dim, 
        fc_hidden_dim=args.fc_hidden_dim, 
        output_dim=1, 
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout
    ).to(device)
    
    pbm_simulator = PROCESS_BASED_MODEL(config_path=args.config_path)
    
    # These parameters are now optimized by Optuna, but we set a default here for direct runs.
    pbm_params = {
        'diff_params': 3e-9,
        'k0_n2o': 0.06,
        'init_on': 2.5,
        'ph': 7.75
    }
    
    # Training
    optimizer = optim.Adam(pinn.parameters(), lr=args.learning_rate)
    trained_model, history = train_model(
        model=pinn,
        pbm_simulator=pbm_simulator,
        pbm_params=pbm_params,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        lambda_pbm=args.lambda_pbm,
        patience=args.patience,
        y_scaler=y_scaler,
        device=device,
        output_dir=run_dir 
    )

    logger.info("Evaluating the final model on the test set...")
    evaluate_model(trained_model, test_loader, device, output_dir=run_dir, y_scaler=y_scaler)


if __name__ == "__main__":
    main()