import torch
import torch.nn as nn 

class DOC(nn.Module):
    def __init__(self, num_cont_var=4, embedding_dim=2, hidden_size=128, seq_len=32, latent_dim=32, num_layers=2, bidirectional=True):
        super(DOC, self).__init__()

        self.dow_embed = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.holiday_embed = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=2*embedding_dim + num_cont_var, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.in_dim = (1+bidirectional)*hidden_size
        self.fc_out = nn.Linear(in_features=self.in_dim, out_features=latent_dim)

    def forward(self, x_cont, x_cat):

        x_dow = x_cat[:, :,0].int()
        x_holiday = x_cat[:, :,1].int()

        dow_embedded = self.dow_embed(x_dow)
        holiday_embedded = self.holiday_embed(x_holiday)

        concatenated = torch.cat([x_cont, dow_embedded, holiday_embedded], dim=2)
        output, (hidden, cell) = self.lstm(concatenated)
        output = self.fc_out(output)

        return output[:, -1, :]