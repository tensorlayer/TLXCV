from typing import Any, Callable, Optional, Tuple

from tensorlayerx.files import load_cifar10_dataset

from .vision import VisionDataset


class Cifar10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset.
        split (string, optional): The image split to use. Can be one of ``train`` (default), ``test``.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that takes in an image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        x_train, y_train, x_test, y_test = load_cifar10_dataset(path=root)
        if split == 'train':
            self.images = x_train
            self.targets = y_train
        else:
            self.images = x_test
            self.targets = y_test

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        image = self.images[index]
        target = self.targets[index]
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.images)
