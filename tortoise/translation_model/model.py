import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1d(ni, no, ks=5):
    return nn.Sequential(
        nn.Conv1d(ni, no, ks, padding=(ks-1)//2, stride=1),
        nn.InstanceNorm1d(no),
        nn.GELU()
    )

def conv2d(ni, no, ks=5):
    return nn.Sequential(
        nn.Conv2d(ni, no, ks, padding=(ks-1)//2, stride=1),
        nn.InstanceNorm2d(no),
        nn.GELU()
    )

class ResBlock(nn.Module):
    def __init__(self, nc=100):
        super().__init__()
        
        self.conv1d_branch = nn.Sequential(
            conv1d(100, 256),
            conv1d(256, 100)
        )
        
        self.conv2d_branch = nn.Sequential(
            conv2d(1, 32),
            conv2d(32, 32),
            conv2d(32, 1, ks=1)
        )
        
    def forward(self, x):
        x_c1d = self.conv1d_branch(x)
        x_c2d = self.conv2d_branch(x[:, None])[:, 0]
        
        return x + x_c1d + x_c2d
    

class Decoder(nn.Module):
    def __init__(self, text_lat_size=1024, embedding_dim=100, num_decoder_layers=1):
        super().__init__()
        
        self.channel_down = nn.Linear(text_lat_size, embedding_dim, bias=True)
        self.pos_enc_size = embedding_dim
        
        self.decoders = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=4, 
                                        dim_feedforward=embedding_dim, batch_first=True) 
              for _ in range(num_decoder_layers)]
        )

    def pos_enc(self, pos, pos_enc_size=100):
        wk = lambda k: (10000)/ (10000**(k/(pos_enc_size//2)))
        embs = [[torch.sin(wk(k) * pos), torch.cos(wk(k) * pos)] for k in range(1, pos_enc_size//2+1)]
        embs_flat = [a for l in embs for a in l]
        return torch.stack(embs_flat, dim=-1)  # bs, L, 100
    
    def forward(self, x, text_lats):
        inp = x
        
        pos = torch.linspace(0, 1, x.shape[-1])[None].expand(x.shape[0], -1)  # bs, L
        pos_enc = self.pos_enc(pos).to(x.device)  # bs, L, 100
        inp = inp.permute(0,2,1) + pos_enc
        
        pos = torch.linspace(0, 1, text_lats.shape[1])[None].expand(text_lats.shape[0], -1)  # bs, L1
        pos_enc = self.pos_enc(pos, inp.shape[-1]).to(x.device)  # bs, L1, 100
        text_lats = self.channel_down(text_lats) + pos_enc
        
        
        for decoder in self.decoders:
            inp = decoder(inp, text_lats)
        
        return inp.permute(0,2,1) + x
    
class TranslationModel(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        
        self.resblocks = nn.ModuleList(
            [nn.Sequential(ResBlock(), ResBlock()) for _ in range(num_layers)]
        )
        self.decoders =  nn.ModuleList(
            [Decoder() for _ in range(num_layers)]
        )
        
        self.gamma = nn.Parameter(torch.tensor([1.0]).float())
        self.final_conv = nn.Sequential(
            nn.Conv1d(200, 1024, 5, padding=2, bias=True),
            nn.InstanceNorm1d(1024),
            nn.GELU(),
            nn.Conv1d(1024, 200, 5, padding=2, bias=True),
            nn.InstanceNorm1d(200),
            nn.GELU(),
            nn.Conv1d(200, 100, 3, padding=1, bias=False),
        )
        # weight = (torch.eye(100) + torch.randn(100, 100)/10)[..., None]
        # self.final_conv[-1].weight = nn.Parameter(weight)
      
    def tacotron_rescale(self, x):
        TACOTRON_MEL_MAX = 2.3143386840820312
        TACOTRON_MEL_MIN = -11.512925148010254
        
        return x * (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)/2 + (TACOTRON_MEL_MAX + TACOTRON_MEL_MIN)/2
    
    def forward(self, x, text_lats):
        inp = x
        for resblock, decoder in zip(self.resblocks, self.decoders):
            inp_conv = resblock(inp)
            inp = inp_conv + inp
            inp = decoder(inp, text_lats)
        # return self.final_conv(self.gamma * inp + x)
        # return self.gamma * inp + self.final_conv(x)
        # return self.tacotron_rescale(torch.tanh(self.final_conv(inp + x)))
        # return self.tacotron_rescale(torch.tanh(self.final_conv(inp)))
        return torch.tanh(self.final_conv(torch.cat([inp, x], dim=1)))