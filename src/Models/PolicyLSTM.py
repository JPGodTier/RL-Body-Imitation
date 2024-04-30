import torch
import torch.nn as nn

class PolicyLSTM(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=128, output_dim=13, num_layers=2):
        """
        Class of a Policy LSTM model that takes a skeleton as input and outputs the angles of the joints.

        Args:
            input_dim (int, optional): Dimension of the skeleton as input. Defaults to 17 points * 3 coordinates = 51.
            hidden_dim (int, optional): Number of features in the hidden state. Defaults to 128. Can be changed.
            output_dim (int, optional): Predicting one angle per joint. Defaults to 13.
            num_layers (int, optional): Number of stacked LSTM layers. Defaults to 2. Can be changed.
        """
        super(PolicyLSTM, self).__init__()
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        
        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)
        
        # Activation function to ensure output is between -1 and 1
        self.activation = nn.Tanh()

    def forward(self, x):
        # Forward pass through LSTM layer
        # x shape: (batch_size, sequence_length, input_dim)
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # Only take the output from the last timestep
        lstm_out = lstm_out[:, -1, :]
        
        # Pass the output of the last LSTM cell to the output layer
        linear_output = self.linear(lstm_out)
        
        # Apply the tanh activation function
        output = self.activation(linear_output)
        
        return output
