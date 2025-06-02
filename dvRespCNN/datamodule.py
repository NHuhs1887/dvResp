from torch.utils.data import DataLoader
import pytorch_lightning as pl
from aedat_dataset import AEDATRespirationDataset
import torch

class AEDATDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/042025",
                 csv_path="data/ground_truth.csv",
                 batch_size=8,
                 frames_per_sample = 150,
                filtered = True):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.filtered = filtered

    def setup(self, stage=None):
        self.dataset = AEDATRespirationDataset(
            data_dir=self.data_dir,
            csv_path=self.csv_path,
            sensor_size=(640,480,2),
            frames_per_sample=150,
            filtered=self.filtered
        )
        total_size = len(self.dataset)
        val_size = int(0.2 * total_size)
        test_size = int(0.1 * total_size)
        train_size = total_size - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # for reproducibility
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,num_workers=0)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                        batch_size=self.batch_size,num_workers=0)
