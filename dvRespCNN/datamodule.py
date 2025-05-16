from torch.utils.data import DataLoader
import pytorch_lightning as pl
from aedat_dataset import AEDATRespirationDataset

class AEDATDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/042025",
                 csv_path="data/ground_truth.csv",
                 batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = AEDATRespirationDataset(
            data_dir=self.data_dir,
            csv_path=self.csv_path,
            sensor_size=(640,480,2),
            frames_per_sample=32
        )
        # optionally split into train/val here

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size)
