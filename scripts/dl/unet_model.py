import torch
import torch.nn as nn

class UNet(nn.Module):
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

        self.d1 = CBR(in_channels, 64)
        self.d2 = CBR(64, 128)
        self.d3 = CBR(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.u2 = CBR(256+128, 128)
        self.u1 = CBR(128+64, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))

        u2 = self.up(c3)
        u2 = self.u2(torch.cat([u2, c2], dim=1))

        u1 = self.up(u2)
        u1 = self.u1(torch.cat([u1, c1], dim=1))

        return torch.sigmoid(self.out(u1))