import yaml

import numpy as np
import torch
import torch.nn as nn
from models.DGCNN_PAConv import PAConv
from models.Diffusion import *
with open('./config/ModelNet40_for_PAConv.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
print("cfg type,  ", type(cfg))

data = np.random.random((16, 3, 8000))
data = torch.from_numpy(data).float()

model = PAConv(cfg.get("MODEL"))
y = model(data)

net = PointWiseNet(50, True)
var_sche = VarianceSchedule(100, 1e-4, 0.02)
diffusion = DiffusionPoint(net, var_sche)

out = diffusion(data.permute(0, 2, 1), y, 50)

print(type(out))