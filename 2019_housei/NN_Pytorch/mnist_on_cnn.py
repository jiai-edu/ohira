"""
## 参考
- [examples/main.py at master · pytorch/examples](https://github.com/pytorch/examples/blob/master/mnist/main.py)
    - 変数名などはちょこちょこ変更している
    - argparseは関係ないので使わない
    - cudaも使わない
    - 手順をわかりやすくするためにいろいろ関数にしている
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 引数を意識するために明示的に書いている
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=20,
                               kernel_size=5,
                               stride=1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(in_features=4 * 4 * 50, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)  # Flatten Layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def prepare_train_loader(my_transform: transforms.Compose) -> DataLoader:
    mnist_train = datasets.MNIST(root='../input',
                                 train=True,
                                 download=True,
                                 transform=my_transform)

    # 試行錯誤しやすくするために意図的にデータ数を減らす
    mnist_train.data = mnist_train.data[:1000]
    train_loader = DataLoader(mnist_train,
                              batch_size=100,
                              shuffle=True)
    return train_loader


def prepare_test_loader(my_transform: transforms.Compose) -> DataLoader:
    mnist_test = datasets.MNIST(root='../input',
                                train=False,
                                download=True,
                                transform=my_transform)
    # 試行錯誤しやすくするために意図的にデータ数を減らす
    mnist_test.data = mnist_test.data[:3000]

    test_loader = DataLoader(mnist_test,
                             batch_size=100,
                             shuffle=True)
    return test_loader


def train(epoch: int, model: Net, optimizer: optim, train_loader: DataLoader) -> None:
    # TODO: Training mode ってなんだ？
    model.train()

    for batch_index, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print(model.conv1.weight.grad.data[0][0][0]) # 適当な重みの変化を観察した

        if batch_index % 2 == 0:
            log_train_msg = f'Train epoch: {epoch} [{batch_index * len(data)}/{len(train_loader.dataset)}]' \
                f'({100. * batch_index / len(train_loader):.0f}%)' \
                f'\tLoss: {loss.item():.6f}'

            print(log_train_msg)


def test(model: Net, test_loader: DataLoader) -> None:
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)

            test_loss += F.nll_loss(output,
                                    target,
                                    reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_log_msg = f'\nTest set: Average loss: {test_loss:.4f}, ' \
        f'Accuracy: {correct}/{len(test_loader.dataset)}' \
        f'({100. * correct / len(test_loader.dataset):.0f}%)\n'

    print(test_log_msg)


def main():
    torch.manual_seed(0)

    my_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = prepare_train_loader(my_transform)
    test_loader = prepare_test_loader(my_transform)

    model = Net()

    optimizer = optim.Adam(params=model.parameters(), lr=0.01)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer, train_loader)
        test(model, test_loader)


if __name__ == "__main__":
    main()
