import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from tensorlayerx.vision import load_image

from .vision import VisionDataset


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class CasiaWebFace(VisionDataset):
    def __init__(
        self,
        root: str,
        extensions: Optional[Tuple[str, ...]] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        百度云链接: https://pan.baidu.com/s/1cnnKrYQDheNfoEhcDoShyA 提取码:vk36
        """
        super().__init__(root, transforms, transform, target_transform)

        self.extensions = extensions or IMG_EXTENSIONS
        self.classes, self.class_to_idx = find_classes(self.root)
        self.samples = samples = self.make_dataset(self.root)
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, label = self.samples[index]
        image = load_image(path)
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label

    def __len__(self) -> int:
        return len(self.samples)

    def is_valid_file(self, x: str) -> bool:
        return x.lower().endswith(self.extensions)

    def make_dataset(
        self,
        directory: str,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        directory = os.path.expanduser(directory)
        class_to_idx = self.class_to_idx
        if not is_valid_file:
            is_valid_file = self.is_valid_file

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if is_valid_file(fname):
                        path = os.path.join(root, fname)
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            msg += f"Supported extensions are: {', '.join(self.extensions)}"
            print("WARN:", msg)

        return instances


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
