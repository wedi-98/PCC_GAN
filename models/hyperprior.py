import torch
import torch.nn as nn

class AnalysisNet(nn.Module):
    def __init__(self, args : dict):
        super(AnalysisNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=9, stride=2, padding_mode='replicate')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding_mode='replicate')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding_mode='replicate')
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return self.conv3(x)

class SynthesisNet(nn.Module):
    def __init__(self, args):
        super(SynthesisNet, self).__init__()
        self.args = args
        self.conv1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding_mode='replicate')
        self.conv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding_mode='replicate')
        self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=50, kernel_size=3, stride=2, padding_mode='replicate')
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return self.act(x)
