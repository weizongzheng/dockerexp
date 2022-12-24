import torch
from torch import nn
from torchvision import *
from torch.utils.data import DataLoader
from fl_model import Net

if __name__ == '__main__':
    net = Net()
    net.load_state_dict(torch.load("./final_model_pars_202202221805541959211"))
    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))
    data_loader_train = DataLoader(mnist_data, batch_size=5, shuffle=False)
    number = len(mnist_data)
    net.eval()  # 测试状态
    total_test_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    total_accuracy = 0  # 正确率
    with torch.no_grad():
        for data in data_loader_train:
            imgs, targets = data
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            # 优化器优化
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试机loss：{}".format(total_test_loss))
    print("整体测试机正确率：{}".format(total_accuracy / number))