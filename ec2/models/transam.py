import torch 
import torch.nn as nn 
from models.revin import RevIN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = 1 / (10000**(2*torch.arange(d_model)/d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position*div_term[0::2])
        pe[:, 1::2] = torch.cos(position*div_term[1::2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.size(1), :]
        return pe + x
    
class TransAm(nn.Module):
    def __init__(self, embedding_dim=2, num_cont_var=2, proj_size=250, num_layers=1, dropout=0.1, revin=False):
        super(TransAm, self).__init__()

        self.revin = revin
        if self.revin: self.revin_layer = RevIN(1, affine=True, subtract_last=False)

        feature_size = embedding_dim + num_cont_var

        self.dow_embed = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)

        self.proj = nn.Linear(feature_size, proj_size)

        self.pos_encoder = PositionalEncoding(d_model=proj_size)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=proj_size, nhead=10, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(in_features=proj_size, out_features=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x_cont, x_cat):

        if self.revin: 
            amount = x_cont[:,:,0].unsqueeze(2)
            amount = self.revin_layer(amount, 'norm')
            x_cont[:, :, 0] = amount.squeeze(-1)

        x_dow = x_cat[:, :,0].int()

        dow_embedded = self.dow_embed(x_dow)

        concatenated = torch.cat([x_cont, dow_embedded], dim=2)

        projection = self.proj(concatenated)

        embedded = self.pos_encoder(projection)

        mask = self._generate_square_subsequent_mask(concatenated.size(1)).to(concatenated.device)

        output = self.transformer_encoder(embedded, mask)
        output = self.decoder(output)
        output = output[:,-1, :].unsqueeze(1)

        if self.revin:
            output = self.revin_layer(output, "denorm")

        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.tril(torch.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask