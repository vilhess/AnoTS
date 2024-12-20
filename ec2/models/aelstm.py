import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, num_cont_var=2, embedding_dim=2, hidden_size=128, num_layers=1, bidirectional=False):
        super(Encoder, self).__init__()

        self.dow_embed = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim + num_cont_var, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x_cont, x_cat):

        x_dow = x_cat[:, :,0].int()

        dow_embedded = self.dow_embed(x_dow)

        concatenated = torch.cat([x_cont, dow_embedded], dim=2)
        output, (hidden, cell) = self.lstm(concatenated)
        return concatenated[:, -1, :], output, (hidden, cell)
    

class BottleNeck(nn.Module):
    def __init__(self, num_cont_var=4, embedding_dim=4, hidden_size=128, latent_size=20):
        super(BottleNeck, self).__init__()
        
        self.fc1_out = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc2_out = nn.Linear(in_features=latent_size, out_features=hidden_size)

        self.fc1_h = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc2_h = nn.Linear(in_features=latent_size, out_features=embedding_dim + num_cont_var)

        self.fc1_c = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.fc2_c = nn.Linear(in_features=latent_size, out_features=embedding_dim + num_cont_var)

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
    def __init__(self, num_cont_var=4, embedding_dim=4, hidden_size=128, num_layers=1, bidirectional=False):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(input_size=hidden_size,  hidden_size=embedding_dim + num_cont_var, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, hidden):

        output, (hidden, cell) = self.lstm(x, hidden)
        return output[:, -1, :]
    

class AELSTM(nn.Module):
    def __init__(self, num_cont_var=2, embedding_dim=2, hidden_size=128, latent_dim=64):
        super(AELSTM, self).__init__()

        self.encoder = Encoder(num_cont_var=num_cont_var, embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.bottleneck = BottleNeck(num_cont_var=num_cont_var, embedding_dim=embedding_dim, hidden_size=hidden_size, latent_size=latent_dim)
        self.decoder = Decoder(num_cont_var=num_cont_var, embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=1, bidirectional=False)

    def forward(self, x_cont, x_cat):
        concatenated, output, (hidden, cell) = self.encoder(x_cont, x_cat)
        output, (hidden, cell) = self.bottleneck(output, (hidden, cell))
        output = self.decoder(output, (hidden, cell))
        return concatenated, output