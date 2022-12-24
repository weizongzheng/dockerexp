import torch
from pfl.core.client import FLClient
from torch.utils.data import Dataset, DataLoader
from pfl.core.trainer_controller import TrainerController
from pfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy
import pandas as pd
import numpy as np
SERVER_URL = "http://127.0.0.1:9763"
CLIENT_IP = "127.0.0.1"
CLIENT_PORT = 8085
CLIENT_ID = 2

from fl_model import Net

from fl_model import Mydata

if __name__ == "__main__":
    root_dir = r".\2018-01-05.csv"


    datasets = Mydata(root_dir)
    client = FLClient()
    pfl_models = client.get_remote_pfl_models(SERVER_URL)

    for pfl_model in pfl_models:
        optimizer = torch.optim.SGD(pfl_model.get_model().parameters(), lr=0.01, momentum=0.5)
        train_strategy = TrainStrategy(optimizer=optimizer, batch_size=1, loss_function=LossStrategy.MSE_LOSS)
        pfl_model.set_train_strategy(train_strategy)

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_CLUSTER, models=pfl_models, data=datasets, client_id=CLIENT_ID,
                      client_ip=CLIENT_IP, client_port=CLIENT_PORT,
                      server_url=SERVER_URL, curve=True, concurrent_num=3).start()
