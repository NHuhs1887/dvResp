# train.py

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping
from model import EventRespirationNet
from aedat_dataset import AEDATRespirationDataset
from datamodule import AEDATDataModule 
import matplotlib.pyplot as plt


def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("loss_curve.png")
    
    
def plot_predictions(y_true, y_pred, title="Predicted vs Ground Truth RR"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.6, label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth RR')
    plt.ylabel('Predicted RR')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    verbose=True
)

class LossHistory(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss.item())

def main():
    # 1) Instantiate model
    model = EventRespirationNet(lr=1e-4)
    loss_history_cb = LossHistory()

    # 2) Instantiate DataModule pointing at your AEDAT folder and GT CSV
    data_module = AEDATDataModule(
        data_dir="data/042025",
        csv_path="data/ground_truth.csv",
        batch_size=1
    )

    # 3) Trainer
    trainer = Trainer(
        max_epochs=50,
        callbacks=[loss_history_cb, early_stop],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10
    )

    # 4) Fit
    trainer.fit(model, datamodule=data_module)
    
    plot_loss_curves(loss_history_cb.train_losses, loss_history_cb.val_losses)
    


if __name__ == "__main__":
    main()
