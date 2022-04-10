import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

INPUT_SIZE = (224, 224)


class RiceDataset(Dataset):  # supposed to run from code directory
    def __init__(self, data_dir, transform=None):
        self.rand = random.Random(42)
        self.data_dir = data_dir
        self.classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        self.samples = self.get_samples()
        self.transform = transform

    def get_samples(self):
        ret = []
        for cls in self.classes:
            for path in (self.data_dir / cls).iterdir():
                ret.append((path, cls))

        self.rand.shuffle(ret)

        return ret

    def __getitem__(self, idx):
        sample_path, cls = self.samples[idx]
        img = Image.open(str(sample_path))
        if self.transform:
            img = self.transform(img)
        cls_idx = self.classes.index(cls)

        return img, cls_idx

    def __len__(self):
        return len(self.samples)


def get_transform():
    transform = transforms.Compose([transforms.Resize(INPUT_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
    return transform
