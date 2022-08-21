import torch
import torch.nn as nn


class InstanceNorm(nn.Module):

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class ApplyStyle(nn.Module):

    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x


class ResnetBlockAdain(nn.Module):

    def __init__(
            self,
            dim,
            latent_size,
            padding_type,
            activation=nn.ReLU(True),
    ):
        super().__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                f'padding [{padding_type}] is not implemented'
            )
        conv1 += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()
        ]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                f'padding [{padding_type}] is not implemented'
            )
        conv2 += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()
        ]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class GeneratorAdainUpsample(nn.Module):

    def __init__(
        self,
        input_nc,
        output_nc,
        latent_size,
        n_blocks=6,
        deep=False,
        norm_layer=nn.BatchNorm2d,
        padding_type='reflect',
    ):
        assert (n_blocks >= 0)
        super().__init__()

        activation = nn.ReLU(True)

        self.deep = deep

        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_nc,
                64,
                kernel_size=7,
            ),
            norm_layer(64),
            activation,
        )
        # downsample
        self.down1 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            norm_layer(128),
            activation,
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1),
            norm_layer(256),
            activation,
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            norm_layer(512),
            activation,
        )

        if self.deep:
            self.down4 = nn.Sequential(
                nn.Conv2d(
                    512,
                    512,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                norm_layer(512),
                activation,
            )

        # resnet blocks
        BN = []
        for _ in range(n_blocks):
            BN += [
                ResnetBlockAdain(512, latent_size, padding_type, activation)
            ]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False,
                ),
                nn.Conv2d(
                    512,
                    512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(512),
                activation,
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            activation,
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            activation,
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )
        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
        )

    def forward(self, input, dlatents):
        x = input

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)
        bot = []
        bot.append(x)
        features = []
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
            bot.append(x)

        if self.deep:
            x = self.up4(x)
            features.append(x)
        x = self.up3(x)
        features.append(x)
        x = self.up2(x)
        features.append(x)
        x = self.up1(x)
        features.append(x)
        x = self.last_layer(x)
        return x