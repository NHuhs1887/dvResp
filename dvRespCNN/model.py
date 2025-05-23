import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

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

class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 3, 3), stride=(1,2,2), padding=(2,1,1)):
        super().__init__()
        kt, kh, kw = kernel_size
        st, sh, sw = stride
        pt, ph, pw = padding

        # temporal conv: across time dimension
        self.temporal = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kt, 1, 1),
            stride=(st, 1, 1),
            padding=(pt, 0, 0)
        )
        # spatial conv: across height and width
        self.spatial = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(1, kh, kw),
            stride=(1, sh, sw),
            padding=(0, ph, pw)
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        return self.relu(x)


class EventRespirationNet(pl.LightningModule):
    def __init__(self, in_channels=2, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.val_preds = []
        self.val_targets = []


        # Encoder: series of 2+1D convs
        self.encoder = nn.Sequential(
            Conv2Plus1D(in_channels, 16),
            Conv2Plus1D(16, 32),
            nn.MaxPool3d((1, 2, 2)),
            Conv2Plus1D(32, 64),
            nn.MaxPool3d((1, 2, 2)),
            Conv2Plus1D(64, 128),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = x.permute(0, 3, 2, 4, 1)
        #print(x.shape())
        x = self.encoder(x)
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss,on_step=False, on_epoch=True, prog_bar=True)
        self.val_preds.append(y_hat.detach().cpu())
        self.val_targets.append(y.detach().cpu())

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_targets = []
    
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
        #plot_predictions(targets.numpy(), preds.numpy(), title="Validation: Predicted vs GT")
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5e-2)
        scheduler = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',               # minimize val_loss
            factor=0.5,               # reduce LR by half
            patience=5,               # wait 5 epochs before reducing
            min_lr=1e-6,              # do not go below this LR
            verbose=True
        ),
        "monitor": "val_loss",       # what metric to monitor
        "interval": "epoch",
        "frequency": 1
    }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    # quick sanity test
    model = EventRespirationNet(in_channels=2)
    dummy = torch.randn(4, 2, 32, 64, 64)
    out = model(dummy)
    print(out.shape)  # should be (4, 1)
