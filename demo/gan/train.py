import os

# NOTE: need to set backend before `import tensorlayerx`
os.environ["TL_BACKEND"] = "paddle"

data_format = "channels_first"
data_format_short = "CHW"

import time

import tensorlayerx as tlx
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import (Compose, Normalize, Resize,
                                            RgbToGray, ToTensor)

from tlxcv.datasets import Cifar10
from tlxcv.models import DCGANModel
from tlxcv.tasks import GAN


class GANTrainer(tlx.model.Model):
    def pd_train(
        self,
        n_epoch,
        train_dataset,
        network,
        loss_fn,
        train_weights,
        optimizer,
        metrics,
        print_train_batch,
        print_freq,
        test_dataset,
    ):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:

            n_batch = len(train_dataset)
            epoch_tqdm = progress.add_task(
                description="[red]Epoch progress 0/{}".format(n_epoch), total=n_epoch
            )
            batch_tqdm = progress.add_task(
                description="[green]Batch progress 0/{}".format(n_batch), total=n_batch
            )

            for epoch in range(n_epoch):
                start_time = time.time()

                train_loss, train_acc, n_iter = 0, 0, 0
                for batch, (X_batch, y_batch) in enumerate(train_dataset):
                    network.backbone.netD.set_train()
                    network.backbone.netG.set_train()
                    output = network(X_batch)

                    network.backbone.netD.set_train()
                    network.backbone.netG.set_eval()
                    netd_loss = network.backbone.backward_D(
                        X_batch, output.detach())
                    grads = optimizer["netD"].gradient(
                        netd_loss, network.backbone.netD.trainable_weights
                    )
                    optimizer["netD"].apply_gradients(
                        zip(grads, network.backbone.netD.trainable_weights)
                    )

                    network.backbone.netD.set_eval()
                    network.backbone.netG.set_train()
                    netg_loss = network.backbone.backward_G(output)
                    grads = optimizer["netG"].gradient(
                        netg_loss, network.backbone.netG.trainable_weights
                    )
                    optimizer["netG"].apply_gradients(
                        zip(grads, network.backbone.netG.trainable_weights)
                    )

                    train_loss += netg_loss.numpy()
                    if metrics:
                        metrics.update(output, y_batch)
                        train_acc += metrics.result()
                        metrics.reset()
                    n_iter += 1

                    if print_train_batch:
                        print(
                            "Epoch {} of {} took {}".format(
                                epoch + 1, n_epoch, time.time() - start_time
                            )
                        )
                        print("   train loss: {}".format(train_loss / n_iter))
                        print("   train acc:  {}".format(train_acc / n_iter))
                    progress.advance(batch_tqdm, advance=1)
                    progress.update(
                        batch_tqdm,
                        description="[green]Batch progress {}/{}".format(
                            batch + 1, n_batch
                        ),
                    )

                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:

                    print(
                        "Epoch {} of {} took {}".format(
                            epoch + 1, n_epoch, time.time() - start_time
                        )
                    )
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

                progress.update(
                    epoch_tqdm,
                    description="[red]Epoch progress {}/{}".format(
                        epoch + 1, n_epoch),
                )
                progress.advance(epoch_tqdm, advance=1)
                progress.reset(batch_tqdm)


if __name__ == "__main__":
    transform = Compose(
        [
            RgbToGray(),
            Resize((64, 64)),
            Normalize(mean=(127.5,), std=(127.5,)),
            ToTensor(data_format=data_format_short),
        ]
    )
    train_dataset = Cifar10(root="./data/cifar10",
                            split="train", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32)

    generator = {
        "input_nz": 100,
        "input_nc": 1,
        "output_nc": 1,
        "ngf": 64,
    }
    discriminator = {
        "ndf": 64,
        "input_nc": 1,
    }
    backbone = DCGANModel(
        generator=generator,
        discriminator=discriminator,
        data_foramt=data_format
    )
    model = GAN(backbone=backbone)

    optimizer = {
        "netG": tlx.optimizers.Adam(0.0001, beta_1=0.5),
        "netD": tlx.optimizers.Adam(0.0001, beta_1=0.5),
    }
    metric = None
    n_epoch = 200

    trainer = GANTrainer(
        network=model,
        loss_fn=model.loss_fn,
        optimizer=optimizer,
        metrics=metric
    )
    trainer.train(
        n_epoch=n_epoch,
        train_dataset=train_dataloader,
        test_dataset=None,
        print_freq=1,
        print_train_batch=False,
    )

    model.save_weights("./demo/gan/model.npz")
