import torch
import torch.nn as nn 

class DOC(nn.Module):
    def __init__(self, input_dim=123, hidden_size=256, latent_dim=32, num_layers=3, bidirectional=True):
        super(DOC, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, bias=False)
        self.in_dim = (1+bidirectional)*hidden_size
        self.fc_out = nn.Linear(in_features=self.in_dim, out_features=latent_dim, bias=False)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.fc_out(output)

        return output[:, -1, :]