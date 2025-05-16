# train.py

from pytorch_lightning import Trainer
from model import EventRespirationNet
from aedat_dataset import AEDATRespirationDataset
from datamodule import AEDATDataModule  # assuming you put it in datamodule.py

def main():
    # 1) Instantiate model
    model = EventRespirationNet(lr=1e-3)

    # 2) Instantiate DataModule pointing at your AEDAT folder and GT CSV
    data_module = AEDATDataModule(
        data_dir="data/042025",
        csv_path="data/ground_truth.csv",
        batch_size=1
    )

    # 3) Trainer
    trainer = Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10
    )

    # 4) Fit
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
