import torch
import torch.nn as nn 
from models.revin import RevIN

class LSTM(nn.Module):
    def __init__(self, num_cont_var=2, embedding_dim=2, hidden_size=128, num_layers=2, bidirectional=True, revin=False):
        super(LSTM, self).__init__()
    
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(1, affine=True, subtract_last=False)

        self.dow_embed = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim + num_cont_var, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc_out = nn.Linear(in_features=(1+bidirectional)*hidden_size, out_features=1)

    def forward(self, x_cont, x_cat):

        if self.revin: 
            amount = x_cont[:,:,0].unsqueeze(2)
            amount = self.revin_layer(amount, 'norm')
            x_cont[:, :, 0] = amount.squeeze(-1)

        x_dow = x_cat[:, :,0].int()
        dow_embedded = self.dow_embed(x_dow)

        concatenated = torch.cat([x_cont, dow_embedded], dim=2)
        output, (hidden, cell) = self.lstm(concatenated)
        output = self.fc_out(output)
        output = output[:, -1, :].unsqueeze(1)

        if self.revin:
            output = self.revin_layer(output, "denorm")
            
        return output