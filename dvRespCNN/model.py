import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1):
        super().__init__()
        kt, kh, kw = kernel_size
        st = stride
        pt = padding

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
            stride=(1, st, st),
            padding=(0, pt, pt)
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        return self.relu(x)


class EventRespirationNet(pl.LightningModule):
    def __init__(self, in_channels=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = x.permute(0, 3, 2, 4, 1)
        x = self.encoder(x)
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    # quick sanity test
    model = EventRespirationNet(in_channels=2)
    dummy = torch.randn(4, 2, 32, 64, 64)
    out = model(dummy)
    print(out.shape)  # should be (4, 1)
