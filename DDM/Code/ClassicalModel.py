import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 定義 VAE 損失函數類
class VAELoss(nn.Module):
    def __init__(self, kl_weight: float = 0.00025):
        super().__init__()
        self.kl_weight = kl_weight
        self.reconstruction_loss_fn = nn.MSELoss(reduction='sum') 

    def forward(self, x_recon, x, mu, logvar):
        recon_loss = self.reconstruction_loss_fn(x_recon, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.kl_weight * kl_divergence

        return total_loss

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=[1,2,3], keepdim=True)
        var = x.var(dim=[1,2,3], keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class AdaGN(nn.Module):
    def __init__(self, embedding_channels, out_channels, num_groups=8):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, out_channels, eps=1e-6, affine=False)
        
        self.fc = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_channels, out_channels * 2)
        )

    def forward(self, x, emb):
        """
        x: [Batch, Channels, H, W]
        emb: [Batch, embedding_channels]
        """

        x = self.groupnorm(x)

        condition = self.fc(emb)

        condition = condition.unsqueeze(-1).unsqueeze(-1)
        gamma, beta = condition.chunk(2, dim=1)

        return x * (1 + gamma) + beta

class ResidualBlockOld(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, Layer_Norm = False):
        super().__init__()

        if Layer_Norm == True:
            self.block1 = nn.Sequential(LayerNorm2d(in_channels), 
                                    nn.SiLU(inplace=True), 
                                    nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
        else:
            self.block1 = nn.Sequential(nn.GroupNorm(num_groups, in_channels),
                                    nn.SiLU(inplace=True), 
                                    nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
            
        if Layer_Norm == True:
            self.block2 = nn.Sequential(LayerNorm2d(out_channels), 
                                               nn.SiLU(inplace=True), 
                                               nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))
        else:
            self.block2 = nn.Sequential(nn.GroupNorm(num_groups, out_channels),
                                    nn.SiLU(inplace=True), 
                                    nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, gamma_beta = None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        
        return h + self.shortcut(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, num_groups=8):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # AdaGN 
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels * 2)
        )

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.silu = nn.SiLU(inplace=True)

        self.shortcut = nn.Identity() if in_channels == out_channels else \
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        nn.init.zeros_(self.emb_layers[-1].weight)
        nn.init.zeros_(self.emb_layers[-1].bias)

    def forward(self, x, emb):
        """
        x: [B, C, H, W]
        emb: [B, emb_channels] 
        """
        h = self.norm1(x)
        h = self.silu(h)
        h = self.conv1(h)

        # --- AdaGN ---
        # [B, out_channels * 2] -> [B, out_channels * 2, 1, 1]
        condition = self.emb_layers(emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = condition.chunk(2, dim=1)
        
        # Normalization
        h = self.norm2(h)
        h = h * (1 + gamma) + beta  # 核心 AdaGN 公式
        
        h = self.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

# VAE 編碼器 (Encoder)
class VAE_Encoder(nn.Module):
    def __init__(self, in_channels = 1, latent_channels = 1, initial_channels=128):
        super().__init__()
        
        # 輸入：[batch_size,in_channel,256,256]
        self.conv_in = nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1, bias=False)
        
        curr_channels = initial_channels
        
        self.block1 = nn.Sequential(ResidualBlockOld(curr_channels, curr_channels),
                                    nn.Conv2d(curr_channels, curr_channels * 2, kernel_size=3, stride=2, padding=1, bias=False)) 
        curr_channels *= 2 # 256

        self.block2 = nn.Sequential(ResidualBlockOld(curr_channels, curr_channels),
                                    nn.Conv2d(curr_channels, curr_channels * 2, kernel_size=3, stride=2, padding=1, bias=False))
        curr_channels *= 2
        
        self.block3 = nn.Sequential(ResidualBlockOld(curr_channels, curr_channels),
                                    nn.GroupNorm(32, curr_channels),
                                    nn.Conv2d(curr_channels, latent_channels*2, kernel_size=3, padding=1, bias=False),
                                    LayerNorm2d(latent_channels*2)) 

    def forward(self, x, V = True):
        x = self.conv_in(x) # in_channel -> initial_channels
        x = self.block1(x)  # [B, initial_channels, 256, 256] -> [B, initial_channels*2, 128, 128]
        x = self.block2(x)  # [B, initial_channels*2, 128, 128] -> [B, initial_channels*4, 64, 64]
        x = self.block3(x)  # [B, initial_channels*4, 64, 64] -> [B, latent_channels*2, 64, 64] 

        mu, logvar = torch.chunk(x, 2, dim=1)

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        if V == True:
            z = mu + epsilon * std
        else:
            z = mu
        return z, mu, logvar

# VAE 解碼器 (Decoder)
class VAE_Decoder(nn.Module):
    def __init__(self, out_channels = 1, latent_channels = 1, initial_channels=512):
        super().__init__()
        
        curr_channels = initial_channels

        self.conv_in = nn.Conv2d(latent_channels, curr_channels, kernel_size=3, padding=1, bias=False)
        
        self.block3 = nn.Sequential(ResidualBlockOld(curr_channels, curr_channels),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Conv2d(curr_channels, curr_channels // 2, kernel_size=3, padding=1, bias=False))
        curr_channels //= 2 # 256
        self.block2 = nn.Sequential(ResidualBlockOld(curr_channels, curr_channels),
                                    nn.Upsample(scale_factor=2, mode='nearest'),
                                    nn.Conv2d(curr_channels, curr_channels // 2, kernel_size=3, padding=1, bias=False))
        curr_channels //= 2 # 128

        self.block1 = nn.Sequential(ResidualBlockOld(curr_channels, curr_channels),
                                    nn.GroupNorm(32, curr_channels),
                                    nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        x = self.conv_in(x) # latent_channels -> initial_channels
        x = self.block3(x)  # [B, initial_channels, 64, 64] -> [B, initial_channels//2, 128, 128]
        x = self.block2(x)  # [B, initial_channels//2, 128, 128] -> [B, initial_channels//4, 256, 256]
        x = self.block1(x)  # [B, initial_channels//4, 256, 256] -> [B, out_channels, 256, 256]

        return torch.tanh(x) 

class VAE_ForDiffusionModel(nn.Module):
    def __init__(self, in_channels = 1, latent_channels = 1, initial_channels = 128):
        super().__init__()
        out_channels = in_channels
        initial_channels = initial_channels
        self.encoder = VAE_Encoder(in_channels = 1, latent_channels=1, initial_channels=initial_channels)
        self.decoder = VAE_Decoder(out_channels = 1, latent_channels=1, initial_channels=initial_channels*4)

    def forward(self, x, V = True):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        # 256x256 -> 64x64
        latent, mu, logvar = self.encoder(x, V = V)
        # 64x64 -> 256x256
        recon = self.decoder(latent)
        return recon, latent, mu, logvar