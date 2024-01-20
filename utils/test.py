import modelnet40_for_PAConv as data
import yaml
import argparse
import numpy as np
with open('../config/ModelNet40_for_PAConv.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


# d = data.ModelNet40ForPAConv(cfg["DATASET"])
# n_min, n_max = d.getitem()
# print("min:    ", n_min, "    n_max:    ", n_max)
# print(d.getitem([0, 1, 2, 3]))
