import torch
import metrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from loss import MultiResolutionSTFTLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, T = x.size()
        query = self.query(x).view(batch_size, -1, T).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, T)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        value = self.value(x).view(batch_size, -1, T)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, T)

        out = self.gamma * out + x
        return out


class DilatedResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, alpha=0.2):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size, padding=2 * (kernel_size - 1) // 2, dilation=2),
            nn.BatchNorm1d(output_channel),
            nn.LeakyReLU(alpha, inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size, padding=4 * (kernel_size - 1) // 2, dilation=4),
            nn.BatchNorm1d(output_channel),
            nn.LeakyReLU(alpha, inplace=True)
        )
        self.in_conv = nn.Conv1d(input_channel, output_channel, kernel_size, padding=kernel_size // 2)

    def forward(self, inputs):
        skip = self.in_conv(inputs)
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = x + skip
        return x
    
class DownSample(nn.Module):
    def __init__(self, factor: int = 2):
        super().__init__()
        self.downsample = nn.MaxPool1d(factor, factor)

    def forward(self, inputs):
        return self.downsample(inputs)


class UpSample(nn.Module):
    def __init__(self, factor: int = 2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=factor, mode='linear', align_corners=True)

    def forward(self, inputs):
        return self.upsample(inputs)



class GLUBlock(nn.Module):
    def __init__(self, n_channels, dilation_rate):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(n_channels, n_channels // 2, kernel_size=1, dilation=1),
            nn.BatchNorm1d(n_channels // 2)
        )
        self.padding = nn.ConstantPad1d((int(dilation_rate * 10), 0), value=0.)
        self.conv_left = nn.Sequential(
            nn.PReLU(),
            self.padding,
            nn.Conv1d(n_channels // 2, n_channels // 2, kernel_size=11, dilation=dilation_rate),
            nn.BatchNorm1d(n_channels // 2)
        )
        self.conv_right = nn.Sequential(
            nn.PReLU(),
            self.padding,
            nn.Conv1d(n_channels // 2, n_channels // 2, kernel_size=11, dilation=dilation_rate),
            nn.BatchNorm1d(n_channels // 2)
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(n_channels // 2, n_channels, kernel_size=1, dilation=1),
            nn.BatchNorm1d(n_channels)
        )
        self.out_prelu = nn.PReLU()

    def forward(self, inputs):
        x = self.in_conv(inputs)
        xl = self.conv_left(x)
        xr = self.conv_right(x)
        x = xl * torch.sigmoid(xr)
        x = self.out_conv(x)
        x = self.out_prelu(x + inputs)
        return x


class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Conv1d expects (B, C, T), LSTM expects (B, T, C)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x.permute(0, 2, 1)


class Generator(nn.Module):
    def __init__(self, channels=1, lite=True):
        super().__init__()
        self.channels = channels
        self.init_dim = 8 if lite else 16

        self.body = nn.Sequential(
            DilatedResBlock(self.channels, self.init_dim, 11),
            DownSample(),
            DilatedResBlock(self.init_dim, self.init_dim * 2, 11),
            DownSample(),
            DilatedResBlock(self.init_dim * 2, self.init_dim * 4, 11),
            DownSample(),
            DilatedResBlock(self.init_dim * 4, self.init_dim * 8, 11),
            DownSample(),
            GLUBlock(dilation_rate=1, n_channels=self.init_dim * 8),
            AttentionBlock(self.init_dim * 8),  # Adding Attention Block
            GLUBlock(dilation_rate=2, n_channels=self.init_dim * 8),
            LSTMBlock(self.init_dim * 8, hidden_dim=128, num_layers=2),  # Adding LSTM Block
            UpSample(),
            DilatedResBlock(self.init_dim * 8, self.init_dim * 8, 7),
            UpSample(),
            DilatedResBlock(self.init_dim * 8, self.init_dim * 4, 7),
            UpSample(),
            DilatedResBlock(self.init_dim * 4, self.init_dim * 2, 7),
            UpSample(),
            DilatedResBlock(self.init_dim * 2, self.init_dim, 7)
        )

        self.last_conv = nn.Sequential(
            nn.ConvTranspose1d(self.init_dim, 1, 1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        x = self.body(inputs)
        x = self.last_conv(x)
        return x


class HybridModel(pl.LightningModule):
    def __init__(self, channels, lite, packet_dim=320, extra_pred_dim=80, lmbda=100.):
        super().__init__()
        self.generator = Generator(channels=channels, lite=lite)
        self.lmbda = lmbda
        self.mse_loss = F.mse_loss
        self.stft_loss = MultiResolutionSTFTLoss()
        self.packet_dim = packet_dim + packet_dim
        self.pred_dim = packet_dim + extra_pred_dim

    def configure_optimizers(self):
        optimizer_g = torch.optim.RAdam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        scheduler = CosineAnnealingLR(optimizer_g, T_max=50)  # Using CosineAnnealingLR
        return {"optimizer": optimizer_g, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        wav, past, ar_data = batch
        pred = self.forward(past) + ar_data

        mse_loss = self.mse_loss(pred, wav)
        sc_loss, log_loss = self.stft_loss(y_pred=pred.squeeze(1), y_true=wav.squeeze(1))
        spectral_loss = 0.5 * (sc_loss + log_loss)

        tot_loss = self.lmbda * mse_loss + spectral_loss

        self.log('tot_loss', tot_loss, prog_bar=True)
        self.log('mse_loss', mse_loss, prog_bar=True)
        self.log('spectral_loss', spectral_loss, prog_bar=True)

        return tot_loss

    def validation_step(self, batch, batch_idx):
        wav, past, ar_data = batch
        pred = self.forward(past) + ar_data

        val_loss = metrics.nmse(y_pred=pred, y_true=wav)
        packet_val_loss = metrics.nmse(y_pred=pred[..., -self.pred_dim:], y_true=wav[..., -self.pred_dim:])

        self.log('val_loss', val_loss)
        self.log('packet_val_loss', packet_val_loss)

        return val_loss, packet_val_loss

    def test_step(self, batch, batch_idx):
        wav, past, ar_data = batch
        pred = self.forward(past) + ar_data

        loss = metrics.nmse(y_pred=pred[..., -self.packet_dim:], y_true=wav[..., -self.packet_dim:])
        self.log('test_loss', loss)

        return loss
