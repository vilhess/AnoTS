import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim=4, hidden_dim=256, num_cont_var=2, num_layers=2, bidirectional=False):
        super(Generator, self).__init__()
        
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(in_features=(1+bidirectional)*hidden_dim, out_features=num_cont_var)
        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

    def forward(self, z):
        rnn_out, _ = self.lstm(z) 
        return self.linear(rnn_out)
    
class Discriminator(nn.Module):
    def __init__(self, num_cont_var=2, hidden_dim=128, num_layers=2, bidirectional=False):
        super(Discriminator, self).__init__()

        self.lstm = nn.LSTM(input_size=num_cont_var, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=(1+bidirectional)*hidden_dim, out_features=1)

        nn.init.trunc_normal_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight)

    def forward(self, x_cont):
            
        output, (hidden, cell) = self.lstm(x_cont)
        output = self.linear(output)
        return output[:, -1, :]
    

def get_best_latent(gen, data, latent_dim):
    gen.train()

    max_iters = 50

    Z = torch.randn((data.size(0), data.size(1), latent_dim), requires_grad=True, device=data.device)
    optimizer = optim.RMSprop(params=[Z], lr=0.1)
    loss_fn = nn.MSELoss(reduction="none")

    normalize_target = F.normalize(data, dim=1, p=2)

    for _ in range(max_iters):
        optimizer.zero_grad()
        
        generated_samples = gen(Z)
        normalized = F.normalize(generated_samples, dim=1, p=2)

        reconstruction_loss = loss_fn(normalized, normalize_target)
        reconstruction_loss = reconstruction_loss.sum(dim=(1, 2)).mean()

        reconstruction_loss.backward()
        optimizer.step()
    gen.eval()
    return Z