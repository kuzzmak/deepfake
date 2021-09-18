from typing import Union
import torch
import torch.nn as nn

from core.activation import Activation, LeakyReLu


class Conv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 'valid',
        activation: Activation = LeakyReLu(0.01),
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
        )

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        return x


class Upscale(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upscale: int = 2,
        stride: int = 1,
        padding: Union[str, int] = 'same',
        activation: Activation = LeakyReLu(0.01),
    ):
        super().__init__()

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * upscale * upscale,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=activation,
        )
        self.pixelShuffle = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.pixelShuffle(x)
        return x


class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 'same',
        activation: Activation = LeakyReLu(0.01),
        conv_activation: Activation = LeakyReLu(0.01),
    ):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=conv_activation,
        )
        self.activation = activation
        self.conv2 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            activation=conv_activation,
        )

    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.activation(y)
        y = self.conv2(y)
        x = y + x
        x = self.activation(x)
        return x
