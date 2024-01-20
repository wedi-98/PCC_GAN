import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
"""
Hyperprior as proposed in [1]. 

[1] Ballé et. al., "Variational image compression with a scale hyperprior", 
    arXiv:1802.01436 (2018).
    
这篇论文提出的模型是在这里实现，总的计算定义在类Hyperprior或者HyperpriorDLMM中
"""



def get_num_DLMM_channels(C, K=4, params=['mu','scale','mix']):
    """
    C:  Channels of latent representation (L3C uses 5).
    K:  Number of mixture coefficients.
    """
    return C * K * len(params)

class HyperpriorAnalysis(nn.Module):
    def __init__(self, C=50, N=320, activation="relu"):
        super(HyperpriorAnalysis, self).__init__()
        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, padding_mode="reflect")
        self.activation = getattr(F, activation)
        self.n_down_sampling_layers = 2
        # TODO 后续改一下参数
        self.conv1 = nn.Conv1d(C, N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(N, N, **cnn_kwargs)
        self.conv3 = nn.Conv1d(N, N, **cnn_kwargs)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        return x


class HyperpriorSynthesis(nn.Module):
    def __init__(self, C=50, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesis, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation

        self.conv1 = nn.ConvTranspose1d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose1d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose1d(N, C, kernel_size=3, stride=1, padding=1)

        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class HyperpriorSynthesisDLMM(nn.Module):
    """
    Outputs distribution parameters of input latents, conditional on
    hyperlatents, assuming a discrete logistic mixture model.

    C:  Number of output channels 后面调一下
    """
    def __init__(self, C=64, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesisDLMM, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation

        self.conv1 = nn.ConvTranspose1d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose1d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose1d(N, C, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(C, get_num_DLMM_channels(C), kernel_size=1, stride=1)

        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        x = self.conv_out(x)

        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


