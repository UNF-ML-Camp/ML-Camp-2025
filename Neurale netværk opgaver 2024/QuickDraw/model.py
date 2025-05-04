"""Modul til definering af netværksarkitektur"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data._data_static import TEGNINGER, dsize
from options import Hyperparameters

C = len(TEGNINGER)

class Squash(nn.Module):
    """
    Squash function as described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.

    Attributes
    ----------
    eps (float): small value to avoid division by zero

    Methods
    -------
    call(inputs): compute the squash function
    """
    def __init__(self, eps=1e-20):
        super(Squash, self).__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        coef = 1 - 1 / (torch.exp(norm) + self.eps)
        unit = x / (norm + self.eps)
        return coef * unit

class PrimaryCaps(nn.Module):
    """
    Create a primary capsule layer with the methodology described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'. 
    Properties of each capsule s_n are exatracted using a 2D depthwise convolution.
    
    Attributes
    ----------
    in_channels (int): number of input channels
    kernel_size (int): size of the kernel
    num_capsules (int): number of capsules
    dim_capsules (int): dimension of each capsule
    stride (int): stride of the convolution

    Methods
    -------
    call(inputs): compute the primary capsule layer
    """
    def __init__(self, in_channels, kernel_size, capsule_size, stride=1,):
        super(PrimaryCaps, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_capsules, self.dim_capsules = capsule_size
        self.stride = stride

        self.dw_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_capsules * self.dim_capsules,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.squash = Squash()
    
    def forward(self, x):
        x = self.dw_conv2d(x)
        x = x.view(-1, self.num_capsules, self.dim_capsules)  # reshape
        return self.squash(x)

class RoutingCaps(nn.Module):
    """
    Create a routing capsule layer with the methodology described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.
    
    Attributes
    ----------
    in_capsules (tuple): input capsules dimensions
    out_capsules (tuple): output capsules dimensions
    squash (Squash): squash function

    Methods
    -------
    call(inputs)
        compute the routing capsule layer
    """
    def __init__(self, in_capsules, out_capsules):
        super(RoutingCaps, self).__init__()
        self.in_capsules = in_capsules # (N_in, D_in)
        self.out_capsules = out_capsules # (N_out, D_out)
        self.squash = Squash()

        # initialize routing parameters
        self.W = nn.Parameter(torch.Tensor(self.out_capsules[0], self.in_capsules[0], self.in_capsules[1], self.out_capsules[1]))
        nn.init.kaiming_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(self.out_capsules[0], self.in_capsules[0], 1))
    
    def forward(self, x):
        ## prediction vectors
        # ji,kjiz->kjz = k and z broadcast, then ji,ji->j = sum(a*b,axis=1)
        u = torch.einsum("...ji,kjiz->...kjz", x, self.W)  # (batch_size/B, N_out, N_in, D1)

        ## coupling coefficients
        # ij,kj->i = ij,kj->k = sum(matmul(a,a.T),axis=0) != ij,ij->i
        c = torch.einsum("...ij,...kj->...i", u, u)  # (B, N1, N0)
        c = c[..., None]  # (B, N_out, N_in, 1) for bias broadcasting
        c = c / torch.sqrt(torch.tensor(self.out_capsules[1]).float())  # stabilize
        c = F.softmax(c,dim=1) + self.b

        ## new capsules
        s = torch.sum(u * c, dim=-2)  # (B, N_out, D_out)
        return self.squash(s)

class CapsLen(nn.Module):
    """
    Create a capsule length layer with the methodology described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'. 
    Compute the length of each capsule n of a layer l.

    Attributes
    ----------
    eps (float): small value to avoid division by zero
    
    Methods
    -------
    call(inputs)
        compute the capsule length layer
    """
    def __init__(self, eps=1e-7):
        super(CapsLen, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.sqrt(
            torch.sum(x**2, dim=-1) + self.eps
        ) # (batch_size, num_capsules)

class Net(nn.Module):
    """
    Netværksarkitektur for klassifikation af billeder
    
    Args:
    nn.Module: Superklasse for alle neurale netværk i PyTorch
    
    Returns:
    Net: Netværksarkitektur
    """
    def __init__(
            self,
            name: str,
            hyperparameters: Hyperparameters,
            input_size=(1, 28, 28),
            n_classes = C,
            n_capsules = 32,
        ):
        # Initialiserer architecturen
        super(Net, self).__init__()

        # Navngiv model
        self.name = name

        # Load Hyperparametre
        self.hyperparameters = hyperparameters

        # Vælg loss function
        self.criterion = nn.CrossEntropyLoss()
        setattr(self.hyperparameters, 'loss', self.criterion.__class__.__name__)

        # Definer lagene i netværket
        self.conv1 = nn.Conv2d(
            in_channels=input_size[0], out_channels=32, kernel_size=5, padding="valid",
        )
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding="valid")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding="valid")
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding="valid")
        self.bn4 = nn.BatchNorm2d(128)

        self.primary_caps = PrimaryCaps(
            in_channels=128, kernel_size=9, capsule_size=(n_capsules, 8)
        )
        self.routing_caps = RoutingCaps(in_capsules=(n_capsules, 8), out_capsules=(n_classes, 16))
        self.len_final_caps = CapsLen()
        self.init_parameters()


    def init_parameters(self):
        """Initialize parameters with Kaiming normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        

    def forward(self, x: torch.Tensor):
        """
        Forward pass af netværket
        
        Args:
        x (torch.Tensor): Input tensor
        
        Returns:
        torch.Tensor: Output tensor
        """
        # Ensure correct shape
        x = x.reshape(-1, 1, 28, 28)

        # forward pass
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.primary_caps(x)
        x = self.routing_caps(x)
        x = self.len_final_caps(x)
        return x
    
    def predict(self, x: torch.Tensor):
        """
        Forudsig klasse
        
        Args:
        x (torch.Tensor): Input data
        
        Returns:
        nd.array: Forudsiget klasse
        nd.array: Forudsiget sandsynlighed
        """
        # Ensure correct shape
        x = x.reshape(-1, 1, 28, 28)

        # Forudsig klasse
        y_hat_prob = self(x)
        y_hat = torch.argmax(y_hat_prob, dim=1)

        return y_hat.detach().numpy()[0], y_hat_prob[0].detach().numpy()
    
    def save(self, path: str = None):
        """
        Gemmer modellen
        
        Args:
        path (str): Sti til gemmested
        """
        # Håndter sti
        if not path:
            if os.path.exists("saved_models"):
                path = f'saved_models/{self.name}.pth'
            elif os.path.exists("QuickDraw/saved_models"):
                path = f'QuickDraw/saved_models/{self.name}.pth'
            else:
                path = f'2.NN/QuickDraw/saved_models/{self.name}.pth'
        scripted_model = torch.jit.script(self)
    
        scripted_model.save(path)

    