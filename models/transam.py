import torch 
import torch.nn as nn 

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
    def __init__(self, embedding_dim=2, num_cont_var=4, proj_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()

        feature_size = 2*embedding_dim + num_cont_var

        self.dow_embed = nn.Embedding(num_embeddings=7, embedding_dim=embedding_dim)
        self.holiday_embed = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

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

        x_dow = x_cat[:, :,0].int()
        x_holiday = x_cat[:, :,1].int()

        dow_embedded = self.dow_embed(x_dow)
        holiday_embedded = self.holiday_embed(x_holiday)

        concatenated = torch.cat([x_cont, dow_embedded, holiday_embedded], dim=2)

        projection = self.proj(concatenated)

        embedded = self.pos_encoder(projection)

        mask = self._generate_square_subsequent_mask(concatenated.size(1)).to(concatenated.device)

        output = self.transformer_encoder(embedded, mask)
        output = self.decoder(output)
        return output[:,-1, :]
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.tril(torch.ones(sz, sz)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask