import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class HAM10000Dataset(Dataset):
    """
    Generic dataset loader based on manifest.csv
    - manifest.csv should have at least: image_path, label, split
    - Optionally: age, sex, site (can be extended later)
    """

    def __init__(self, manifest_csv, split="train", img_size=224):
        self.df = pd.read_csv(manifest_csv)
        if "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)   # normalize to [-1,1]
        ])

        # map string labels to numeric indices
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, label


def make_dataloaders(manifest_csv, batch_size=32, img_size=224, num_workers=2):
    """
    Helper to build train/val/test dataloaders from a manifest.csv
    """
    splits = ["train", "val", "test"]
    loaders = {}

    for split in splits:
        dataset = HAM10000Dataset(manifest_csv, split=split, img_size=img_size)
        shuffle = True if split == "train" else False
        loaders[split] = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=num_workers)
    return loaders
