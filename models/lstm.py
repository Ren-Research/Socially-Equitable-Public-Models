import torch
import torch.nn as nn

class LSTM(nn.Module):
    # def __init__(self, input_size=1, hidden_size=50, output_size=1):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)

        self.softplus = nn.Softplus()

    def forward(self, input_seq):
        # lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        # predictions = self.linear(lstm_out.view(len(input_seq), -1))
        #
        # return predictions[-1]
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        predictions = self.softplus(predictions)
        return predictions.squeeze(-1)