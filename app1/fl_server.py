import torch
from pfl.core.server import FLClusterServer
from pfl.core.strategy import FederateStrategy
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

FEDERATE_STRATEGY = FederateStrategy.FED_AVG
IP = '127.0.0.1'
PORT = 9763
API_VERSION = '/api/version'
class mydata(Dataset):
    def __init__(self, root_dir):  # ,label_dir#放置路径
        d = np.array(pd.read_csv(root_dir).iloc[:,1:])
        data = torch.tensor(d).to(torch.float32)
        window = 1
        label_index = 1
        self.z = []
        for i in range(len(data) - window):
            # 以之前所有时刻数据来预测下一时刻数据
            x = (data[i:(i + window), label_index:])
            y = (data[i + window, label_index])
            self.z.append((torch.tensor(x), torch.tensor(y)))

    def __getitem__(self, idx):  # 获取元素
        d = self.z[idx]
        return d[0],d[1]

    def __len__(self):
        return len(self.z)
if __name__ == "__main__":
    server = FLClusterServer(FEDERATE_STRATEGY, IP, PORT, API_VERSION)
    server.start()
