import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def plot_predictions(y_true, y_pred, title="Predicted vs Ground Truth RR"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.6, label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth RR')
    plt.ylabel('Predicted RR')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("preds.png")
    plt.show()

class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels,med_channels, out_channels, kernel_size=(10, 3, 3), stride=(1,2,2), padding=(0,1,1)):
        super().__init__()
        kt, kh, kw = kernel_size
        st, sh, sw = stride
        pt, ph, pw = padding

        self.conv_t = nn.Sequential(
            nn.Conv3d(in_channels, med_channels, 
                           (kt, 1, 1), stride=(st, 1, 1), padding=(pt, 0, 0), bias=False), 
            nn.Sequential(nn.GroupNorm(4, med_channels), nn.ReLU())
        )
        
        self.conv_s = nn.Sequential(
            nn.Conv3d(med_channels, out_channels, 
                      (1, kh, kw), (1,sh, sw), (1,ph, pw), 
                      bias=False), 
            nn.Sequential(nn.BatchNorm3d(out_channels), nn.ReLU()), 
        )

    def forward(self, x):
        #print("conv_t input:", x.shape)
        x = self.conv_t(x)
        #print("conv_t output / conv_s input:", x.shape)
        return self.conv_s(x)


class EventRespirationNet(pl.LightningModule):
    def __init__(self, in_channels=2, lr=1e-4, kernel_size=(10, 3, 3)):
        super().__init__()
        self.save_hyperparameters()
        self.val_preds = []
        self.val_targets = []
        self.lstm_outputs = []
        self.kernel_size=kernel_size


        # Modified encoder with smaller strides and fewer pools
        self.encoder = nn.Sequential(
            Conv2Plus1D(in_channels, 8, 16, stride=(1, 1, 1), kernel_size=self.kernel_size),  
            Conv2Plus1D(16, 16, 32,kernel_size=self.kernel_size),                            
            Conv2Plus1D(32, 16, 32),                           
            Conv2Plus1D(32, 32, 64),                            
            Conv2Plus1D(64, 32, 64),                            
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            Conv2Plus1D(64, 64, 128),                          
            nn.ReLU(),
            Conv2Plus1D(128, 64, 128, stride=(1, 1, 1)),        
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 4, 4))
        )

        # Decoder now upsamples both T and spatial dims
        #self.decoder = nn.Sequential(
        #    nn.ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # Upsample T, H, W
        #    nn.BatchNorm3d(64),
        #    nn.ReLU(),
        #)

        self.lstm = nn.LSTM(input_size=2048,  # Assume H=12, W=16 → adjust based on actual output
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.3)

        #self.fc = nn.Linear(64 * 2, 1) * 4 * 4,  # = 512


        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x, return_embedding=False):
        # x shape: (B, C, T, H, W)
        #print("INput shape: " + str(x.shape))
        x = x.permute(0, 3, 2, 4, 1)
        #print("after permute: " + str(x.shape))
        x = self.encoder(x)
        #print("after encode: " + str(x.shape))
        # (B, C, T, H, W) → (B, T, C * H * W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.flatten(2)              # (B, T, C*H*W)
        #print("after reshape: " + str(x.shape))
        lstm_out, _ = self.lstm(x)    # → (B, 1, 128)
        #print("after lstm: " + str(x.shape))
        x = lstm_out[:, -1, :]   
        #print("after lstm2: " + str(x.shape))
        if return_embedding:
            return x  # return LSTM embedding only
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.huber_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        lstm_embed = self(x, return_embedding=True)
        y_hat = self(x).squeeze(1)
        loss = F.huber_loss(y_hat, y)
        self.log("val_loss", loss,on_step=False, on_epoch=True, prog_bar=True)
        self.val_preds.append(y_hat.detach().cpu())
        self.val_targets.append(y.detach().cpu())
        self.lstm_outputs.append(lstm_embed.detach().cpu())

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_targets = []
        self.lstm_outputs = []
    
    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
        #plot_predictions(targets.numpy(), preds.numpy(), title="Validation: Predicted vs GT")
        #self.inspect_output_layer()
        #self.plot_lstm_embeddings() 
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.huber_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
        # Save predictions and targets
        self.test_preds.append(y_hat.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        
        return loss
    
    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy()
        targets = torch.cat(self.test_targets).numpy()
    
        # Compute metrics
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        r2 = r2_score(targets, preds)
    
        # Log metrics
        self.log("test_mae", mae, prog_bar=True)
        self.log("test_rmse", rmse, prog_bar=True)
        self.log("test_r2", r2, prog_bar=True)
    
        # Plot predictions
        plot_predictions(targets, preds, title="Test: Predicted vs Ground Truth RR")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',               # minimize val_loss
            factor=0.5,               # reduce LR by half
            patience=5,               # wait 5 epochs before reducing
            min_lr=1e-10,              # do not go below this LR
            verbose=True
        ),
        "monitor": "val_loss",       # what metric to monitor
        "interval": "epoch",
        "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "gradient_clip_val": 1.0,}

    def inspect_output_layer(self):
        with torch.no_grad():
            print("encoder:")
            for name, param in self.encoder.named_parameters():
                print(f"{name} - mean: {param.mean().item():.4f}, std: {param.std().item():.4f}")

                
    def plot_lstm_embeddings(self):
    
        embeddings = torch.cat(self.lstm_outputs).numpy()  # (N, D)
        plt.figure(figsize=(10, 6))
        plt.hist(embeddings.flatten(), bins=100, color='skyblue', edgecolor='black')
        plt.title("Histogram of LSTM Output Embeddings")
        plt.xlabel("Embedding Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    model = EventRespirationNet(in_channels=2)
    dummy = torch.randn(4, 2, 32, 64, 64)
    out = model(dummy)
    print(out.shape)  # should be (4, 1)
