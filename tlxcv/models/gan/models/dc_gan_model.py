import tensorlayerx as tlx
import tensorlayerx.nn as nn
from tensorlayerx.losses import binary_cross_entropy

from .discriminators import DCDiscriminator
from .generators import DCGenerator


class DCGANModel(nn.Module):
    """
    This class implements the DCGAN model, for learning a distribution from input images.
    DCGAN paper: https://arxiv.org/pdf/1511.06434
    """

    def __init__(self, generator, discriminator=None, data_foramt="channels_first"):
        """Initialize the DCGAN class.
        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
        """
        super().__init__()
        self.input_nz = generator["input_nz"]

        # define networks (both generator and discriminator)
        self.netG = DCGenerator(data_foramt=data_foramt, **generator)
        if discriminator:
            self.netD = DCDiscriminator(
                data_format=data_foramt, **discriminator)

    def forward(self, input):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""

        # generate random noise and fake image
        z = tlx.random_normal(
            shape=(tlx.get_tensor_shape(input)[0], self.input_nz, 1, 1)
        )
        fake = self.netG(z)
        return fake

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Args:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            if not hasattr(self, "target_real_tensor"):
                self.target_real_tensor = tlx.constant(
                    value=1.0,
                    shape=tlx.get_tensor_shape(prediction),
                    dtype=tlx.float32,
                )
            target_tensor = self.target_real_tensor
        else:
            if not hasattr(self, "target_fake_tensor"):
                self.target_fake_tensor = tlx.constant(
                    value=0.0,
                    shape=tlx.get_tensor_shape(prediction),
                    dtype=tlx.float32,
                )
            target_tensor = self.target_fake_tensor

        return target_tensor

    def loss_fn(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Args:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)[
            :tlx.get_tensor_shape(prediction)[0]]
        loss = binary_cross_entropy(prediction, target_tensor)
        return loss

    def backward_D(self, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = self.netD(fake)
        loss_D_fake = self.loss_fn(pred_fake, False)

        pred_real = self.netD(real)
        loss_D_real = self.loss_fn(pred_real, True)

        # combine loss and calculate gradients
        loss = (loss_D_fake + loss_D_real) * 0.5
        return loss

    def backward_G(self, fake):
        """Calculate GAN loss for the generator"""
        # G(A) should fake the discriminator
        pred_fake = self.netD(fake)
        return self.loss_fn(pred_fake, True)
