import torch
from pfl.core.client import FLClient
from torchvision import datasets, transforms
from pfl.core.trainer_controller import TrainerController
from pfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy

SERVER_URL = "http://127.0.0.1:9763"
CLIENT_IP = "127.0.0.1"
CLIENT_PORT = 8084
CLIENT_ID = 1



if __name__ == "__main__":
    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))

    client = FLClient()
    pfl_models = client.get_remote_pfl_models(SERVER_URL)

    for pfl_model in pfl_models:
        optimizer = torch.optim.SGD(pfl_model.get_model().parameters(), lr=0.01, momentum=0.5)
        train_strategy = TrainStrategy(optimizer=optimizer, batch_size=32, loss_function=LossStrategy.NLL_LOSS)
        pfl_model.set_train_strategy(train_strategy)

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_CLUSTER, models=pfl_models, data=mnist_data, client_id=CLIENT_ID,
                      client_ip=CLIENT_IP, client_port=CLIENT_PORT,
                      server_url=SERVER_URL, curve=True, concurrent_num=3).start()
