# pinn_model.py
import torch.nn as nn

class PINN_Model(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, fc_hidden_dim, output_dim, num_lstm_layers, dropout=0.2):
        super(PINN_Model, self).__init__()
        
        # LSTM Layer
        # It processes sequences of data and captures temporal dependencies.
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,  # This makes handling batch dimensions easier
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Feedforward Head
        # This part takes the output of the LSTM and makes the final prediction.
        self.fc_network = nn.Sequential(
            nn.Linear(lstm_hidden_dim, fc_hidden_dim),
            nn.ReLU(), # Activation function to introduce non-linearity 
            nn.BatchNorm1d(fc_hidden_dim), # Normalization for better training stability
            nn.Dropout(dropout), # Dropout for regularization and to prevent overfitting
            nn.Linear(fc_hidden_dim, output_dim)
        )

    def forward(self, x):
        # The LSTM returns the output for each time step, plus the final hidden and cell states.
        # We only need the final output of the sequence for our prediction.
        lstm_out, _ = self.lstm(x) # (batch_size, time_length, lstm_hidden_dim)
        
        # Handle the case where the input sequence length is zero
        if lstm_out.size(1) == 0:
            raise ValueError("Input sequence length must be greater than 0.")
        
        # We take the output of the last time step to feed into our fully connected layer.
        final_lstm_out = lstm_out[:, -1, :] # (batch_size, lstm_hidden_dim) for the last time step
        
        # Pass the LSTM output through the feedforward head for the final prediction.
        prediction = self.fc_network(final_lstm_out)
        return prediction