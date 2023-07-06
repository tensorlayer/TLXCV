from typing import Any, Callable, Optional, Tuple

import numpy as np

from .vision import VisionDataset


class Circles(VisionDataset):
    def __init__(
        self,
        num: int,
        nx: int = 172,
        ny: int = 172,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(None, transforms, transform, target_transform)
        self.num = num
        self.nx = nx
        self.ny = ny

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = _create_image_and_mask(self.nx, self.ny)
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label

    def __len__(self) -> int:
        return self.num


def _create_image_and_mask(nx, ny, cnt=10, r_min=3, r_max=10, border=32, sigma=20):
    image = np.ones((nx, ny, 1))
    mask = np.zeros((nx, ny), dtype=np.bool)

    for _ in range(cnt):
        a = np.random.randint(border, nx - border)
        b = np.random.randint(border, ny - border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1, 255)

        y, x = np.ogrid[-a: nx - a, -b: ny - b]
        m = x * x + y * y <= r * r
        mask = np.logical_or(mask, m)

        image[m] = h

    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    mask = np.stack([~mask, mask], axis=-1)

    return image, mask
