import torch
import torch.nn as nn

class EncoderDecoderModel(nn.Module):
    '''
        Execerise 1.1 
        Saptial progression: 128 -> 64 -> 32 -> 64 -> 128
        Channel progression: 2 -> 16 -> 32 -> 128 -> 32 -> 16 -> 1
    '''
    def __init__(self):
        super(EncoderDecoderModel, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), # 2 channels (No blue channel)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(16, 1, kernel_size=1)


    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool1(x)
        x = self.encoder2(x)
        x = self.pool2(x)
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = self.decoder2(x)
        x = self.final(x)
        return x
    
class EncoderDecoderModelV2(nn.Module):
    def __init__(self):
        super(EncoderDecoderModelV2, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (32 -> 32, no expansion)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool1(x)
        x = self.encoder2(x)
        x = self.pool2(x)
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = self.decoder2(x)
        x = self.final(x)
        return x

class EncoderDecoderModelV3(EncoderDecoderModelV2):

    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool1(x)
        x = self.encoder2(x)
        x = self.pool2(x)
        x = self.dropout(x)        # dropout after encoder
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = self.decoder2(x)
        x = self.final(x)
        return x

class UNet(EncoderDecoderModel):
    '''
        Exercise 1.2 — U-Net with skip connections.
    '''
    def __init__(self):
        super().__init__()
        # Override decoder layers to accept concatenated skip connections
        # decoder1 outputs 64ch, encoder2 skip is 32ch → 64+32 = 96 input
        self.decoder1 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # decoder2 outputs 16ch, encoder1 skip is 16ch → 16+16 = 32 input
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.pool1(e1)
        e2 = self.encoder2(x)
        x = self.pool2(e2)
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = torch.cat([x, e2], dim=1)  # skip from encoder2
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = torch.cat([x, e1], dim=1)  # skip from encoder1
        x = self.decoder2(x)
        x = self.final(x)
        return x

class UNetV2(EncoderDecoderModelV2):
    def __init__(self):
        super().__init__()

        self.decoder1 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.pool1(e1)
        e2 = self.encoder2(x)
        x = self.pool2(e2)
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = torch.cat([x, e2], dim=1)  # 16 + 32 = 48
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = torch.cat([x, e1], dim=1)  # 16 + 16 = 32
        x = self.decoder2(x)
        x = self.final(x)
        return x

class UNetV3(UNetV2):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.pool1(e1)
        e2 = self.encoder2(x)
        x = self.pool2(e2)
        x = self.dropout(x)        # dropout after encoder
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = torch.cat([x, e2], dim=1)  # 16 + 32 = 48
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = torch.cat([x, e1], dim=1)  # 16 + 16 = 32
        x = self.decoder2(x)
        x = self.final(x)
        return x

class ResUNet(EncoderDecoderModel):
    '''
    Exercise 1.3 — Like UNet but with additive (sum) skip connections
    instead of concatenation. Channels stay the same.
    '''
    def __init__(self):
        super().__init__()
        # upconv1 must output 32ch to match encoder2 for addition (was 64 in base)
        self.upconv1 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        # decoder1 input is 32 (no channel doubling), already matches base decoder1
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # upconv2 outputs 16, encoder1 is 16 — already matches, no override needed

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.pool1(e1)
        e2 = self.encoder2(x)
        x = self.pool2(e2)
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = x + e2              # additive skip from encoder2
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = x + e1              # additive skip from encoder1
        x = self.decoder2(x)
        x = self.final(x)
        return x

class ResUNetV2(EncoderDecoderModelV2):
    def __init__(self):
        super().__init__()

        # Must output 32ch to match e2 for addition
        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        # upconv2 and decoder2 unchanged — 16ch matches e1 already

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.pool1(e1)
        e2 = self.encoder2(x)
        x = self.pool2(e2)
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = x + e2              # additive skip, both 32ch
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = x + e1              # additive skip, both 16ch
        x = self.decoder2(x)
        x = self.final(x)
        return x

class ResUNetV3(ResUNetV2):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.pool1(e1)
        e2 = self.encoder2(x)
        x = self.pool2(e2)
        x = self.dropout(x)        # dropout after encoder
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = x + e2              # additive skip, both 32ch
        x = self.decoder1(x)
        x = self.upconv2(x)
        x = x + e1              # additive skip, both 16ch
        x = self.decoder2(x)
        x = self.final(x)
        return x