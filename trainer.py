# trainer.py
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER_NAME = "trainer_logger"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d – %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def step(self, val_loss, model):
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_state)

def custom_loss_function(y_true, y_pbm, y_pred, lambda_pbm):
    """ Custom loss function combining data loss and PBM loss."""
    data_loss = F.mse_loss(y_pred, y_true)
    # The PBM loss compares the Process-based model output to the predicted output by the ML model
    pbm_loss = F.mse_loss(y_pred, y_pbm) # Physics-informed loss
    total_loss = data_loss + lambda_pbm * pbm_loss
    return total_loss, data_loss, pbm_loss

def train_model(model, pbm_simulator, pbm_params, train_loader, 
                val_loader, optimizer, num_epochs, lambda_pbm, patience,
                y_scaler, device, output_dir):
    """ Training loop for the PINN model with early stopping."""
    
    # Training Loop with Early Stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch_seq, y_batch in train_loader:
            X_batch_seq, y_batch = X_batch_seq.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            # PINN forward pass with the full sequence
            y_pred = model(X_batch_seq)
            
            # PBM forward pass using only the LAST time step of the sequence
            # Shape: (batch_size, seq_length, features) -> (batch_size, features)
            X_batch_last_step = X_batch_seq[:, -1, :].cpu().numpy()
            y_pbm_raw = pbm_simulator.run_simulation(X_batch_last_step, **pbm_params)
            
            # Scale the PBM output before calculating loss ***
            y_pbm_scaled = torch.tensor(y_scaler.transform(y_pbm_raw), dtype=torch.float32).to(device)


            # Compute combined loss
            total_loss, _, _ = custom_loss_function(y_batch, y_pbm_scaled, y_pred, lambda_pbm)
            # Backward pass to compute derivative of loss function w.r.t. model parameters
            total_loss.backward()
            # Update model parameters
            optimizer.step()
            train_losses.append(total_loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch_seq, y_batch in val_loader:
                X_batch_seq, y_batch = X_batch_seq.to(device), y_batch.to(device)
                
                y_pred = model(X_batch_seq)
                X_batch_last_step = X_batch_seq[:, -1, :].cpu().numpy()
                y_pbm_raw = pbm_simulator.run_simulation(X_batch_last_step, **pbm_params)
                
                # Scale the PBM output for validation loss as well ***
                y_pbm_scaled = torch.tensor(y_scaler.transform(y_pbm_raw), dtype=torch.float32).to(device)
                
                
                total_loss, _, _ = custom_loss_function(y_batch, y_pbm_scaled, y_pred, lambda_pbm)
                val_losses.append(total_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        # Check early stopping condition, if met, break the loop
        early_stopping.step(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break
    
    # Plotting and Saving Loss Curve 
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training and Validation Loss')
    # Save the figure instead of showing it
    plot_path = os.path.join(output_dir, 'loss_vs_epoch.png')
    plt.savefig(plot_path)
    plt.close() # Close the plot to prevent the script from hanging
    logger.info(f"Loss plot saved to {plot_path}")

    return model, history

def evaluate_model(model, test_loader, device, output_dir, y_scaler):
    """Evaluates the model and saves the prediction plot."""
    model.eval()
    y_true_scaled, y_pred_scaled = [], []
    with torch.no_grad():
        for X_batch_seq, y_batch_scaled in test_loader:
            X_batch_seq = X_batch_seq.to(device)
            outputs = model(X_batch_seq)
            y_true_scaled.append(y_batch_scaled.numpy())
            y_pred_scaled.append(outputs.cpu().numpy())
    
    y_true_scaled = np.vstack(y_true_scaled)
    y_pred_scaled = np.vstack(y_pred_scaled)

    # Inverse transform the predictions and true values to original scale ***
    y_true = y_scaler.inverse_transform(y_true_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    logger.info(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    #  Plotting True vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('True Values (mg/m2/d)')
    plt.ylabel('Predicted Values (mg/m2/d)')
    plt.title('True vs. Predicted Values on Test Set')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'true_vs_predicted.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Prediction plot saved to {plot_path}")
    
    return mae, rmse, r2