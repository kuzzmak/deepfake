from typing import Tuple

import torch

from core.activation import LeakyReLu, Sigmoid
from core.layer import Conv2d, Linear, Flatten, ResBlock, Upscale
from core.model.ae_model import DeepfakeAEModel


class OriginalAE(DeepfakeAEModel):
    """Original Deepfake model based on https://github.com/dfaker/df."""

    def __init__(self, input_shape: Tuple[int, int, int]):
        """Constructor.

        Parameters
        ----------
        input_shape : Tuple[int, int, int]
            image shape which model will receive on it's input in format
            C, H, W where C represents number of channels, H represents height
            of the image and W, width of the image
        """
        assert len(input_shape) == 3, 'Wrong image shape passed as an ' + \
            'argument, must have three values C - number of image channels' + \
            ', H - height of the image, W - width of the image.'

        encoder_channels = [
            (input_shape[0], 128),
            (128, 256),
            (256, 512),
            (512, 1024),
        ]

        conv_template = 'conv_{}_{}'

        super().__init__(
            encoder_channels=encoder_channels,
            conv_template=conv_template,
            input_shape=input_shape,
        )

    def init_layers(self):

        lrelu_01 = LeakyReLu(0.1)
        lrelu_02 = LeakyReLu(0.2)
        sigmoid = Sigmoid()

        for channels in self.encoder_channels:
            in_ch, out_ch = channels
            setattr(
                self,
                self.conv_template.format(in_ch, out_ch),
                Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5,
                    stride=2,
                    activation=lrelu_01,
                ),
            )

        # calculate size of the input in linear layer
        last_conv_filters = self.encoder_channels[-1][1]
        linear_in_features = self._calculate_linar_input_size(
            last_conv_filters,
            5,
            2,
        )

        self.flatten = Flatten()
        self.linear_1024 = Linear(linear_in_features, 1024)
        self.linear_16384 = Linear(1024, 4 * 4 * 1024)
        self.upscale_512_e = Upscale(
            in_channels=1024,
            out_channels=512,
            kernel_size=3,
            activation=lrelu_01,
        )

        self.upscale_512_d_1 = Upscale(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            activation=lrelu_01,
        )
        self.res_block_512_d_1 = ResBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            activation=lrelu_02,
        )
        self.upscale_256_d_1 = Upscale(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            activation=lrelu_01,
        )
        self.res_block_256_d_1 = ResBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            activation=lrelu_02,
        )
        self.upscale_128_d_1 = Upscale(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            activation=lrelu_01,
        )
        self.res_block_128_d_1 = ResBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            activation=lrelu_02,
        )
        self.upscale_64_d_1 = Upscale(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            activation=lrelu_01,
        )
        self.conv_d_1 = Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=5,
            padding='same',
            activation=sigmoid,
        )
        self.upscale_512_d_2 = Upscale(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            activation=lrelu_01,
        )
        self.upscale_256_d_2 = Upscale(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            activation=lrelu_01,
        )
        self.upscale_128_d_2 = Upscale(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            activation=lrelu_01,
        )
        self.upscale_64_d_2 = Upscale(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            activation=lrelu_01,
        )
        self.conv_d_2 = Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=5,
            padding='same',
            activation=sigmoid,
        )

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        for channels in self.encoder_channels:
            in_ch, out_ch = channels
            conv = getattr(self, self.conv_template.format(in_ch, out_ch))
            x = conv(x)
        x = self.flatten(x)                     # [bs, linear_in_features]
        x = self.linear_1024(x)                 # [bs, 1024]
        x = self.linear_16384(x)                # [bs, 16384]
        x = torch.reshape(x, (-1, 1024, 4, 4))  # [bs, 1024, 4, 4]
        x = self.upscale_512_e(x)               # [bs, 512, 8, 8]
        return x

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.upscale_512_d_1(x)            # [bs, 512, 16, 16]
        x1 = self.res_block_512_d_1(x1)         # [bs, 512, 16, 16]
        x1 = self.upscale_256_d_1(x1)           # [bs, 256, 32, 32]
        x1 = self.res_block_256_d_1(x1)         # [bs, 256, 32, 32]
        x1 = self.upscale_128_d_1(x1)           # [bs, 128, 64, 64]
        x1 = self.res_block_128_d_1(x1)         # [bs, 128, 64, 64]
        x1 = self.upscale_64_d_1(x1)            # [bs, 64, 128, 128]
        x1 = self.conv_d_1(x1)                  # [bs, 3, 128, 128]

        x2 = self.upscale_512_d_2(x)            # [bs, 512, 16, 16]
        x2 = self.upscale_256_d_2(x2)           # [bs, 256, 32, 32]
        x2 = self.upscale_128_d_2(x2)           # [bs, 128, 64, 64]
        x2 = self.upscale_64_d_2(x2)            # [bs, 64, 128, 128]
        x2 = self.conv_d_2(x2)                  # [bs, 1, 128, 128]

        return x1, x2

    def _calculate_linar_input_size(
        self,
        last_conv_filters: int,
        kernel: int,
        stride: int,
    ):
        shape = [self.input_shape[1], self.input_shape[2]]
        for _ in range(len(self.encoder_channels)):
            w = int((shape[1] - kernel) / stride + 1)
            h = int((shape[0] - kernel) / stride + 1)
            shape[0] = h
            shape[1] = w
        return shape[0] * shape[1] * last_conv_filters
