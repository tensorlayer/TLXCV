import tensorlayerx.nn as nn


class DCDiscriminator(nn.Module):
    """Defines a DCGAN discriminator"""

    def __init__(self, input_nc, ndf=64, data_format="channels_first"):
        """Construct a DCGAN discriminator

        Parameters:
            input_nc (int): the number of channels in input images
            ndf (int): the number of filters in the last conv layer
            norm_type (str): normalization layer type
        """
        super().__init__()

        kw = 4
        padw = 1

        sequence = [
            nn.Conv2d(
                in_channels=input_nc,
                out_channels=ndf,
                kernel_size=kw,
                stride=2,
                padding=padw,
                data_format=data_format,
            ),
            nn.LeakyReLU(0.2),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        n_downsampling = 4

        # gradually increase the number of filters
        for n in range(1, n_downsampling):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    in_channels=ndf * nf_mult_prev,
                    out_channels=ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    data_format=data_format,
                ),
                nn.BatchNorm2d(
                    num_features=ndf * nf_mult,
                    data_format=data_format
                ),
                nn.LeakyReLU(0.2),
            ]

        nf_mult_prev = nf_mult

        # output 1 channel prediction map
        sequence += [
            nn.Conv2d(
                in_channels=ndf * nf_mult_prev,
                out_channels=1,
                kernel_size=kw,
                stride=1,
                padding=0,
                data_format=data_format,
            ),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
