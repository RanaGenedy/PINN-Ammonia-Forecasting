<!-- README.md -->
# Physics-Informed Neural Network (PINN) for Ammonia Emission Forecasting

This project implements a hybrid modeling approach, combining a traditional *Process-Based Model (PBM)* with a *deep learning model (LSTM)* to forecast ammonia (NH₃) emissions from manure storage facilities. This Physics-Informed Neural Network (PINN) leverages the strengths of both data-driven and physics-driven methods.

## Key Features
- **Process-Based Model (PBM):** A detailed, one-dimensional simulation of the physical and biochemical processes in manure storage, including heat transfer, mass transfer, and nitrogen mineralization.
- **LSTM Neural Network:** A Long Short-Term Memory (LSTM) network captures complex temporal dependencies from historical weather and management data.
- **Hybrid PINN Approach:** The LSTM model is trained not only on measured data but also constrained by the output of the PBM. The custom loss function penalizes deviations from both the ground truth and the physics-based predictions.
- **Hyperparameter Optimization:** Includes a script (optuna_optimizer.py) to systematically find the best hyperparameters for both the PBM and the LSTM using the Optuna framework.
- **Modular & Configurable:** The code is structured into logical modules, and experiments can be easily configured and run from the command line.

## Project Structure
```bash
PINN_Project/
│
├── runs/                     # Default directory for saving experiment results
│   └── 2025-09-21_18-41-02/    # Example timestamped run folder
│       ├── loss_vs_epoch.png
│       └── true_vs_predicted.png
│
├── PBM_config.yaml           # Configuration for the Process-Based Model parameters and constants
├── pbm_module.py             # The Process-Based Model (PBM) simulator
├── data_loader.py            # Data loading, scaling, and sequencing logic for input data
├── pinn_model.py             # The PINN (LSTM) model architecture
├── trainer.py                # Training, validation, and evaluation logic 
├── utils.py                  # Utility functions
├── main.py                   # Main script to run a single training instance
├── optuna_optimizer.py       # Script for hyperparameter optimization
├── requirements.txt          # File with all the Libraries required for installation
└── README.md                 # This file
```

## Data for PINN Ammonia Emission Forecasting
**Sample Data**
The file ```dataset_sample.csv``` contains a small, representative sample of the full dataset. It includes the first few hundred rows to allow for immediate testing and validation of the code pipeline.

## Setup and Installation
1. Clone the repository:
```bash
git clone <repo-url>
cd PINN_Project
```
2. Create and activate a virtual environment (recommended):
```bash
python -m venv PINN_env
source PINN_env/bin/activate 
```
3. Install the required packages:
All necessary packages are listed in *requirements.txt* Install them using the following command:
```bash
pip install -r requirements.txt
```

## Configuration
The *PBM_config.yaml* file contains all the physical constants and parameters for the Process-Based Model. You can adjust values like herd size (NAU), manure properties, and storage dimensions in this file without changing the Python code.

## Usage
**1. Running a Single Training Experiment**
Use the main.py script to train and evaluate the model with a specific set of hyperparameters. The only required argument is --data_path.

*Example (using defaults):*
```bash
python main.py --data_path <PATH_TO_DATA>
```
*Example (with custom hyperparameters):*
This command trains for 200 epochs with a smaller learning rate and a different model architecture.
```bash
python main.py \
    --data_path <PATH_TO_DATA> \
    --output_dir results \
    --epochs 200 \
    --learning_rate 0.0005 \
    --lambda_pbm 0.7 \
    --lstm_hidden_dim 64 \
    --num_lstm_layers 1
```
Results and plots will be saved in a timestamped subfolder within the results directory.

**2. Optimizing Hyperparameters with Optuna**
Use the *optuna_optimizer.py* script to automatically search for the best combination of hyperparameters for both the PBM and the LSTM model.
```bash
python optuna_optimizer.py --data_path <PATH_TO_DATA> --n_trials <#> --output_dir <PATH_TO_OUTPUT_DIR>
```
This will run a set number of trials (e.g., 100) and print the best-performing set of parameters at the end.

## Modules Explained
```main.py:``` The main entry point for running a single training experiment. Handles argument parsing and orchestrates the workflow.
```optuna_optimizer.py:``` The entry point for running hyperparameter optimization using Optuna.
```data_loader.py```: Handles loading the CSV data, resampling it to an hourly frequency, scaling features and targets, and creating sequential data loaders suitable for the LSTM.
```pinn_model.py:``` Defines the PINN_LSTM_Model class, which consists of an LSTM layer to capture time-series patterns followed by a feedforward head for prediction.
```pbm_module.py:``` Contains the PROCESS_BASED_MODEL class, which simulates the physics of ammonia emission.
```trainer.py:``` Contains the core logic for training the model, including the custom PINN loss function, validation loop, early stopping, and evaluation metrics.
```utils.py:``` Contains helper functions, such as seed_everything for ensuring reproducibility.
```dataset_sample.csv``` Contains a sample of data for code testing
