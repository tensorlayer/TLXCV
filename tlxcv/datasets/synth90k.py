import random
import os
from tensorlayerx.dataflow import Dataset


class Synth90k(Dataset):
    def __init__(self, archive_path, split="train", transform=None):
        self.archive_path = archive_path
        self.transform = transform

        if split == "train":
            transcripts_file = os.path.join(archive_path, "annotation_train.txt")
        else:
            transcripts_file = os.path.join(archive_path, "annotation_test.txt")

        files = []
        for i in open(transcripts_file):
            i = i.strip().split(" ")
            text = i[0].split("_")[1]
            files.append((i[0], text))

        self.files = files

    def __getitem__(self, index: int):
        jpg_index, text = self.files[index]
        jpg_path = os.path.join(self.archive_path, jpg_index)

        if self.transform:
            try:
                image, target = self.transform(jpg_path, text)
            except Exception:
                print('Error data, removing:', self.files[index])
                del self.files[index]
                index = random.randrange(0, len(self.files))
                return self[index]

        return image, (target, text)

    def __len__(self) -> int:
        return len(self.files)
