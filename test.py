from torch.utils.data import DataLoader
import yaml

from models.DGCNN_PAConv import PAConv
from models.Hyperprior import Hyperprior
import numpy as np
import torch

with open('./config/ModelNet40_for_PAConv.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# data = ModelNet40ForPAConv(cfg.get("DATASET"))
# dataset = DataLoader(data, batch_size=1)
model = PAConv(cfg.get("TRAINING"))
model.train()
# for i, data in enumerate(dataset):
#     data = np.array(data, dtype=np.float64)
#     print(data.shape)
#     data = torch.tensor(data, dtype=torch.float64)
#     y = model(data)
#     print(y.shape)
#     if i == 0:
#         break
data = np.random.random((4, 3, 8000))
data = torch.from_numpy(data).float()
y = model(data)
y = y.permute(0,2,1)
hyper = Hyperprior(bottleneck_capacity=50)
y = hyper(y, (50, 8000))

print(y.size())
