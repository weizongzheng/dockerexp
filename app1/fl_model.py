import math

from torch import nn
import torch.nn.functional as F
import pfl.core.strategy as strategy
from pfl.core.job_manager import JobManager
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
class Mydata(Dataset):
    def __init__(self, root_dir):
        d = np.array(pd.read_csv(root_dir).iloc[:,1:])
        data = torch.tensor(d).to(torch.float32)
        window = 1
        label_index = 0
        self.z = []
        for i in range(len(data) - window):
            if i < 50:
                x = (data[0:i + 1, label_index:])
                y = (data[i + window, label_index:])
                self.z.append((torch.tensor(x), torch.tensor(y)))
            else:
                x = (data[i - 50:i + 1, label_index:])
                y = (data[i + window, label_index:])
                self.z.append((torch.tensor(x), torch.tensor(y)))

    def __getitem__(self, idx):
        d = self.z[idx]
        return d[0],d[1]

    def __len__(self):
        return len(self.z)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class Net(nn.Module):
    def __init__(self, input_size = 4, hidden_size = 161, num_layers = 2):
        super(Net, self).__init__()
        self.model = nn.LSTM(input_size=4, hidden_size= 161, num_layers=2)
        self.line = nn.Linear(hidden_size, 8)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.pos = PositionalEncoding(4)

    def forward(self, input):
        shape = input.shape
        input = input.view(shape[1],shape[0],shape[2])
        #input = input.unsqueeze(1).to(torch.float32)
        #print(input.shape)
        if input.shape[0] > 5:
            out, h = self.model(input[-5:, ...])
            out = out[-1, ...]
            lineOut = self.line(out)
            lineOut = lineOut.view(8)
            mu, sigma = lineOut.chunk(2, dim=0)
            result1 = mu + sigma * torch.randn_like(sigma)
        else:
            out, h = self.model(input)
            out = out[-1, ...]
            lineOut = self.line(out)
            lineOut = lineOut.view(8)
            mu, sigma = lineOut.chunk(2, dim=0)
            result1 = mu + sigma * torch.randn_like(sigma)

        input = self.pos(input)

        result2 = self.transformer_encoder(input)
        result2 = result2[-1, ...]

        result = result1 + result2

        return result



if __name__ == "__main__":
    input_size = 4
    hidden_size = 161
    num_layers = 2
    model = Net()
    job_manager = JobManager()
    job = job_manager.generate_job(work_mode=strategy.WorkModeStrategy.WORKMODE_CLUSTER,
                                   fed_strategy=strategy.FederateStrategy.FED_AVG, epoch=50, model=Net , l2_dist=True)
    job_manager.submit_job(job, model)
