import tensorlayerx.nn as nn


class DCGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations."""

    def __init__(
        self, input_nz, input_nc, output_nc, ngf=64, data_foramt="channels_first"
    ):
        """Construct a DCGenerator generator

        Args:
            input_nz (int): the number of dimension in input noise
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            ngf (int): the number of filters in the last conv layer
            norm_layer: normalization layer
            padding_type (str): the name of padding layer in conv layers: reflect | replicate | zero
        """
        super().__init__()

        mult = 8
        n_downsampling = 4

        model = [
            nn.ConvTranspose2d(
                in_channels=input_nz,
                out_channels=ngf * mult,
                kernel_size=4,
                stride=1,
                padding=0,
                data_format=data_foramt,
            ),
            nn.BatchNorm2d(num_features=ngf * mult, data_format=data_foramt),
            nn.ReLU(),
        ]

        # add upsampling layers
        for i in range(1, n_downsampling):
            mult = 2 ** (n_downsampling - i)
            output_size = 2 ** (i + 2)
            model += [
                nn.ConvTranspose2d(
                    in_channels=ngf * mult,
                    out_channels=ngf * mult // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    data_format=data_foramt,
                ),
                nn.BatchNorm2d(
                    num_features=ngf * mult // 2,
                    data_format=data_foramt
                ),
                nn.ReLU(),
            ]

        output_size = 2 ** (6)
        model += [
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=output_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                data_format=data_foramt,
            ),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward"""
        return self.model(x)
