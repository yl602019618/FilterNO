import pytorch_lightning as pl
import torch
import wandb
import numpy as np

import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.loss import LpLoss,RRMSE
from models.basics_model import get_grid2D,FC_nn
##################
#fourier convolution 2d block
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    

class RRMSE(object):
    def __init__(self, ):
        super(RRMSE, self).__init__()
        
    def __call__(self, x, y):
        num_examples = x.size()[0]
        norm = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), 2 , 1)**2
        normy = torch.norm( y.view(num_examples,-1), 2 , 1)**2
        mean_norm = torch.mean((norm/normy)**(1/2))
        return mean_norm

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x





class FNO2d(pl.LightningModule):
    def __init__(self, modes1 = 20, 
                 modes2 = 20, 
                 width = 20, 
                 padding = 2, 
                 input_dim = 1, 
                 output_dim = 1,
                 loss = "rel_l2",
                    learning_rate = 1e-2, 
                    step_size= 100,
                    gamma= 0.5,
                    weight_decay= 1e-5,
                    eta_min = 5e-4):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_dim+2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)
        if loss == 'l1':
            self.criterion = nn.L1Loss()
            self.criterion_val = LpLoss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            self.criterion_val = LpLoss()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
            self.criterion_val = LpLoss()
        elif loss == "rel_l2":
            self.criterion =LpLoss()
            self.criterion_val = RRMSE()
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.val_iter = 0

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    def training_step(self, batch: torch.Tensor, batch_idx):    
        x,y = batch
        batch_size = x.shape[0]
        #y = y-self.homo_field.unsqueeze(0) # NEW
        out = self(x)
        loss = self.criterion(out.view(batch_size,-1),y.view(batch_size,-1))#torch.mean(torch.abs(out.view(batch_size,-1)-10*y.view(batch_size,-1)) ** 2)
        #loss = torch.mean(torch.abs(out.view(batch_size,-1)- y.view(batch_size,-1)) ** 2)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"loss": loss.item()})
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        self.val_iter += 1
        x,y= val_batch
        batch_size = x.shape[0]
        #out = self(sos,src)+10*self.homo_field.unsqueeze(0) #new
        out = self(x)
        val_loss = self.criterion_val(out.view(batch_size,-1),y.view(batch_size,-1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_loss": val_loss.item()})
        if self.val_iter %10 ==0:
            #self.log_wandb_image(wandb,sos[0].detach().cpu(),(y-self.homo_field.unsqueeze(0))[0].detach().cpu(),(out-10*self.homo_field.unsqueeze(0))[0].detach().cpu())
            self.log_wandb_image(wandb,x[0].detach().cpu(),y[0].detach().cpu(),out[0].detach().cpu())
        return val_loss

    def log_wandb_image(self,wandb,  sos, field, pred_field):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax = ax.flatten()
        ax0 = ax[0].imshow(sos[...,0], cmap="inferno")
        ax[0].set_title("Sound speed")
        ax[1].imshow(field[...,0], cmap="inferno")
        ax[1].set_title("Field")
        ax[2].imshow(pred_field[...,0], cmap="inferno")
        ax[2].set_title("Predicted field")
        img = wandb.Image(plt)
        wandb.log({'Image': img})
        plt.close()
    def configure_optimizers(self, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if  scheduler is None:
            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.step_size, eta_min= self.eta_min)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }
