import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityBlock(nn.Module):
    """
    1x1 (stride=1) -> BN -> ReLU ->
    3x3 (stride=1, padding=1) -> BN -> ReLU ->
    1x1 (stride=1) -> BN
    + shortcut
    -> ReLU
    """
    def __init__(self, in_ch, f1, f2, f3):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_ch, f1, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(f1)
        self.conv2 = nn.ConvTranspose2d(f1,  f2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(f2)
        self.conv3 = nn.ConvTranspose2d(f2,  f3, kernel_size=1, stride=1, bias=False)
        self.bn3   = nn.BatchNorm2d(f3)
        self.relu  = nn.ReLU(inplace=True)

        # in_ch == f3 by design in your graph
        assert in_ch == f3, f"IdentityBlock channel mismatch: in={in_ch}, out={f3}"

    def forward(self, x):
        shortcut = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + shortcut)
        return out


class DeconvBlock(nn.Module):
    """
    1x1 (stride=1) -> BN -> ReLU ->
    3x3 (stride=1, padding=1) -> BN -> ReLU ->
    1x1 (stride=1 or 2) -> BN
    + shortcut (1x1, stride=1 or 2)
    -> ReLU

    When stride=2, use 1x1 ConvTranspose2d as upsampling 。
    """
    def __init__(self, in_ch, f1, f2, f3, stride=1):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride

        self.conv1 = nn.ConvTranspose2d(in_ch, f1, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(f1)

        self.conv2 = nn.ConvTranspose2d(f1,  f2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(f2)


        if stride == 2:
            self.conv3 = nn.ConvTranspose2d(f2, f3, kernel_size=1, stride=2,
                                            output_padding=1, bias=False)
            self.short = nn.ConvTranspose2d(in_ch, f3, kernel_size=1, stride=2,
                                            output_padding=1, bias=False)
        else:
            self.conv3 = nn.ConvTranspose2d(f2, f3, kernel_size=1, stride=1, bias=False)
            self.short = nn.ConvTranspose2d(in_ch, f3, kernel_size=1, stride=1, bias=False)

        self.bn3   = nn.BatchNorm2d(f3)
        self.bn_short = nn.BatchNorm2d(f3)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.bn_short(self.short(x))

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.relu(out + shortcut)
        return out

# ---------------------------
# ResNet-184 
# ---------------------------
class _ResNet184DecoderCore(nn.Module):
    """
    In put ：Linear(latent_dim -> 8*8*2048) → reshape (B,2048,8,8)
    The main structure（conv2_x～conv5_x）：[3, 8, 36, 3, 3, 2]
    Last two stages：deconv=5 → three identity；deconv=6 → two identity（stride=1）
    Out put ：1x1 ConvTranspose 到 out_ch
    Resolution ：256×256
    """
    def __init__(self, latent_dim: int, out_ch: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_ch = out_ch

        # Stem
        self.fc = nn.Linear(latent_dim, 8*8*2048)

        # ===== deconv=1（8x8, 2048）=====
        # identity × rep5=3，stride=2 , upsampling
        self.stage1_id = nn.Sequential(
            IdentityBlock(2048, 512, 512, 2048),
            IdentityBlock(2048, 512, 512, 2048),
            IdentityBlock(2048, 512, 512, 2048),
        )
        self.stage1_up = DeconvBlock(2048, 512, 512, 1024, stride=2)  # -> 16x16

        # ===== deconv=2（16x16, 1024）=====
        # identity × rep4=36， 32x32 / 512
        self.stage2_id = nn.Sequential(*[
            IdentityBlock(1024, 256, 256, 1024) for _ in range(36)
        ])
        self.stage2_up = DeconvBlock(1024, 256, 256, 512, stride=2)   # -> 32x32

        # ===== deconv=3（32x32, 512）=====
        # identity × rep3=8， 64x64 / 256
        self.stage3_id = nn.Sequential(*[
            IdentityBlock(512, 128, 128, 512) for _ in range(8)
        ])
        self.stage3_up = DeconvBlock(512, 128, 128, 256, stride=2)    # -> 64x64

        # ===== deconv=4（64x64, 256）=====
        # identity × rep2=3， 128x128 / 128
        self.stage4_id = nn.Sequential(
            IdentityBlock(256, 64,  64,  256),
            IdentityBlock(256, 64,  64,  256),
            IdentityBlock(256, 64,  64,  256),
        )
        self.stage4_up = DeconvBlock(256, 64, 64, 128, stride=2)      # -> 128x128


        # deconv=5（128x128, 128）：identity × 3， 256x256 / 64
        self.tail5_id = nn.Sequential(
            IdentityBlock(128, 32, 32, 128),
            IdentityBlock(128, 32, 32, 128),
            IdentityBlock(128, 32, 32, 128),
        )
        self.tail5_up = DeconvBlock(128, 32, 32, 64, stride=2)        # -> 256x256

        # deconv=6（256x256, 64）：identity × 2，stride=1 
        self.tail6_id = nn.Sequential(
            IdentityBlock(64, 16, 16, 64),
            IdentityBlock(64, 16, 16, 64),
        )
        self.tail6_align = DeconvBlock(64, 8, 8, 32, stride=1)        # -> 256x256

        # Output head
        self.head = nn.ConvTranspose2d(32, out_ch, kernel_size=1, stride=1)

        self._init_weights()

    def _init_weights(self):
        # Kaiming initial（Suitable ReLU）
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        """
        z: (B, latent_dim) 
        back : (B, out_ch, 256, 256)
        """
        assert z.dim() == 2 and z.size(1) == self.latent_dim, \
            f"Expect z shape (B,{self.latent_dim}), got {tuple(z.shape)}"

        x = self.fc(z)                       # (B, 8*8*2048)
        x = x.view(-1, 2048, 8, 8)          # (B, 2048, 8, 8)

        x = self.stage1_id(x)
        x = self.stage1_up(x)               # 16x16 / 1024

        x = self.stage2_id(x)
        x = self.stage2_up(x)               # 32x32 / 512

        x = self.stage3_id(x)
        x = self.stage3_up(x)               # 64x64 / 256

        x = self.stage4_id(x)
        x = self.stage4_up(x)               # 128x128 / 128

        x = self.tail5_id(x)
        x = self.tail5_up(x)                # 256x256 / 64

        x = self.tail6_id(x)
        x = self.tail6_align(x)             # 256x256 / 32

        x = self.head(x)                    # 256x256 / out_ch
        return x



class ResNet184Decoder1(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.core = _ResNet184DecoderCore(latent_dim, out_ch=1)
    def forward(self, z):
        return self.core(z)

