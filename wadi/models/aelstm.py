import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, input_dim=123, hidden_size=256, num_layers=2, bidirectional=False):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)
    

class BottleNeck(nn.Module):
    def __init__(self, input_dim=123, hidden_size=256, latent_size=20):
        super(BottleNeck, self).__init__()
        
        self.fc1_out = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc2_out = nn.Linear(in_features=latent_size, out_features=hidden_size)

        self.fc1_h = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc2_h = nn.Linear(in_features=latent_size, out_features=input_dim)

        self.fc1_c = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc2_c = nn.Linear(in_features=latent_size, out_features=input_dim)

        self.relu = nn.ReLU()

    def forward(self, output, hidden):
        hidden, cell = hidden

        z_output = self.relu(self.fc1_out(output))
        new_out = self.fc2_out(z_output)

        z_h = self.relu(self.fc1_h(hidden))
        new_h = self.fc2_h(z_h)

        z_c = self.relu(self.fc1_c(cell))
        new_c = self.fc2_c(z_c)

        return new_out, (new_h, new_c)
    
class Decoder(nn.Module):
    def __init__(self, input_dim=123, hidden_size=128, num_layers=1, bidirectional=False):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(input_size=hidden_size,  hidden_size=input_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, hidden):

        output, (hidden, cell) = self.lstm(x, hidden)
        return output[:, -1, :]
    

class AELSTM(nn.Module):
    def __init__(self, input_dim=123, hidden_size=128, latent_dim=64):
        super(AELSTM, self).__init__()

        self.encoder = Encoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.bottleneck = BottleNeck(input_dim=input_dim, hidden_size=hidden_size, latent_size=latent_dim)
        self.decoder = Decoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=1, bidirectional=False)

    def forward(self, x):
        output, (hidden, cell) = self.encoder(x)
        output, (hidden, cell) = self.bottleneck(output, (hidden, cell))
        output = self.decoder(output, (hidden, cell))
        return output