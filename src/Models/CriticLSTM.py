import torch
import torch.nn as nn

class CriticLSTM(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=128, num_layers=2):
        """
        Class of a Critic LSTM model (for Actor-Critic Agent) that takes a skeleton as input and outputs the state Value.

        Args:
            input_dim (int, optional): Dimension of the skeleton as input. Defaults to 17 points * 3 coordinates = 51.
            hidden_dim (int, optional): Number of features in the hidden state. Defaults to 128. Can be changed.
            num_layers (int, optional): Number of stacked LSTM layers. Defaults to 2. Can be changed.
        """
        super(CriticLSTM, self).__init__()
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        
        # Define the output layer
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Forward pass through LSTM layer
        # x shape: (batch_size, sequence_length, input_dim=51)
        # lstm_out shape: (batch_size, sequence_length, hidden_dim=128)
        lstm_out, _ = self.lstm(x)
        
        # Only take the output from the last timestep
        # lstm_out shape: (batch_size, hidden_dim=128)
        lstm_out = lstm_out[:, -1, :]
        
        # Pass the output of the last LSTM cell to the output layer
        # linear_output shape: (batch_size, output_dim=1)
        state_value = self.linear(lstm_out) 
        
        return state_value
