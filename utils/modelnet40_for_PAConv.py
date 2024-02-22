import os
import random
from torch.utils.data import Dataset


class ModelNet40ForPAConv(Dataset):
    def __init__(self, args):
        self.path = args["path"]  # 数据集路径, 该路径下需要包含一个txt文件ModelNet.txt(自己创建)，文件中一行代表包含的一类数据
        self.n = args["n"]  # 保留点的数量
        self.mode = args["mode"]  # 训练模式还是测试模式
        self.txt = os.path.join(self.path, "ModelNet40.txt")
        self.cl_list = []  # 类别列表，对应的文件夹列表
        self.files = []  # 所有文件列表
        with open(self.txt, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                self.cl_list.append(line)
        for cl in self.cl_list:
            files = os.path.join(self.path, cl, "train")
            with open(os.path.join(files, cl + ".txt")) as f:
                for line in f:
                    line = line.replace("\n", "")
                    file = os.path.join(files, line)
                    if os.path.getsize(file) / 1024 > 10:
                        self.files.append(file)
        random.shuffle(self.files)

    def __getitem__(self, item):
        if type(item) != "list":
            item = [item]
        data = []
        for i in item:
            xyzs = []
            file = self.files[i]
            with open(file, "r") as f:
                lines = f.readlines()
                lines = lines[2:]
                for line in lines:
                    line = line.replace("\n", "")
                    line = line.strip()
                    xyz = line.split(" ")
                    if len(xyz) != 3:
                        continue
                    xyz = [float(d) for d in xyz]
                    if len(xyz) != 3:
                        print(len(xyz))
                    xyzs.append(xyz)
            data.append(xyzs)
        return data

    def getitem(self):
        n_min = 200000000
        n_max = 0
        for file in self.files:
            i = 0
            with open(file, "r") as f:
                lines = f.readlines()
                lines = lines[2:]
                for line in lines:
                    line = line.replace("\n", "")
                    line = line.strip()
                    xyz = line.split(" ")
                    if len(xyz) != 3:
                        continue
                    i += 1
                if i < n_min:
                    n_min = i
                if i > n_max:
                    n_max = i
                if i == 0:
                    print(file)
        return n_min, n_max

    def __len__(self):
        return len(self.files)
