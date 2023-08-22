"""Implementation of a Restricted Boltzmann Machine"""
import torch

from utils import *


class RBM():
    """Implementation of a Restricted Boltzmann Machine
    
    Note that this implementation does not use Pytorch's nn.Module
    because we are updating the weights ourselves

    """
    def __init__(self, visible_dim, hidden_dim, gaussian_hidden_distribution=False, Device='cpu'):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.gaussian_hidden_distribution = gaussian_hidden_distribution

        # intialize parameters
        self.W = torch.randn(visible_dim, hidden_dim).to(DEVICE) * 0.1
        self.h_bias = torch.zeros(hidden_dim).to(DEVICE)  # v --> h
        self.v_bias = torch.zeros(visible_dim).to(DEVICE)  # h --> v

        # parameters for learning with momentum
        self.W_momentum = torch.zeros(visible_dim, hidden_dim).to(DEVICE)
        self.h_bias_momentum = torch.zeros(hidden_dim).to(DEVICE)  # v --> h
        self.v_bias_momentum = torch.zeros(visible_dim).to(DEVICE)  # h --> v

    def sample_h(self, v):
        activation = torch.mm(v, self.W) + self.h_bias
        if self.gaussian_hidden_distribution:
            return activation, torch.normal(activation, torch.tensor([1]).to(DEVICE))
        else:
            p = torch.sigmoid(activation)
            return p, torch.bernoulli(p)

    def sample_v(self, h):
        activation = torch.mm(h, self.W.t()) + self.v_bias
        p = torch.sigmoid(activation)
        return p

    def update_weights(self, v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size):
        self.W_momentum *= momentum_coef
        self.W_momentum += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)

        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), 0)

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), 0)

        self.W += lr*self.W_momentum/batch_size
        self.h_bias += lr*self.h_bias_momentum/batch_size
        self.v_bias += lr*self.v_bias_momentum/batch_size

        self.W -= self.W * weight_decay 

