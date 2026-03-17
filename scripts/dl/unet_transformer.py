import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, dim, heads=4, mlp_dim=512):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):

        h = x
        x = self.norm1(x)

        attn_out, _ = self.attn(x, x, x)

        x = h + attn_out

        h = x
        x = self.norm2(x)
        x = h + self.mlp(x)

        return x


class UNetTransformer(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.d1 = CBR(in_channels, 64)
        self.d2 = CBR(64, 128)
        self.d3 = CBR(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Transformer bottleneck
        self.transformer = TransformerBlock(dim=256)

        # Decoder
        self.u2 = CBR(256 + 128, 128)
        self.u1 = CBR(128 + 64, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        # Encoder
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))

        # Transformer bottleneck
        B, C, H, W = c3.shape

        t = c3.flatten(2).transpose(1, 2)   # (B, HW, C)
        t = self.transformer(t)
        c3 = t.transpose(1, 2).reshape(B, C, H, W)

        # Decoder
        u2 = self.up(c3)
        u2 = self.u2(torch.cat([u2, c2], dim=1))

        u1 = self.up(u2)
        u1 = self.u1(torch.cat([u1, c1], dim=1))

        return torch.sigmoid(self.out(u1))