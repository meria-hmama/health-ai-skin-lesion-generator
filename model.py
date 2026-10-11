import torch
import torch.nn as nn
latent_dim = 100

class Generator(nn.Module):                                    #defining a new network called Generator
    def __init__(self):                                        #setup function
        super().__init__()                                      #required boilerplate for PyTorch networks
        self.main = nn.Sequential(                               #stack of layers (running in order)
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),   #turning random noise into a 4x4 feature map
            nn.BatchNorm2d(512),                                 #stabilizing training
            nn.ReLU(True),                                       #activation function

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),   #from 4x4 to 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),   #from 8x8 to 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),    #from 16x16 to 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),      #from 32x32 to 64x64, outputs 3 color channels
            nn.Tanh()                                            # squashing output to [-1, 1] range (matches our normalized real images)
        )

    def forward(self, x):                                        #defining how data flows through the network
        return self.main(x)