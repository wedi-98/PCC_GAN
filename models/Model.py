import torch.nn as nn
from models.DGCNN_PAConv import PAConv
from models.Likelihood import Likelihood

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.feature_extractor = PAConv(args)
        self.likelihood = Likelihood(50)
    def forward(self, x):
        y = self.feature_extractor(x)
        likelihood = self.likelihood(y)
        return likelihood


