import argparse
import timeit
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


'''
本設問はtext classificationではない．
ベクトル(300dim)はラベル中の単語ベクトルの平均となっている
rsync -avzhP spumoni02:/home/SeiyaKikuchi/100knock-2020/SeiyaKikuchi/chapter08/data/result.txt /Users/seiya.k/workspace/100knock-2020/SeiyaKikuchi/chapter08/data/

'''


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="device_id", type=int,
                        help="Serect device id on remote server", default=0)
    return parser


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 75)
        self.fc3 = nn.Linear(75, 4)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # print(x.size())
        x = x.to(dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def train(data_loader, max_epoch, device):
    net = Net()
    net = net.to(device)
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    loss_baseline = 99999.0
    for epoch in range(max_epoch):
        running_loss = 0.0
        for vector, label in data_loader:
            vector, label = vector.to(device), label.to(device)
            optimizer.zero_grad()
            logit = net(vector)
            loss = CELoss(logit, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'epoch:{epoch}, loss:{running_loss}')
        if loss_baseline > running_loss:
            loss_baseline = running_loss
            torch.save(net.state_dict(), 'data/model_param')

    return net


def calc_accuracy(data_loader, device):
    preds = []
    labels = []
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load('data/model_param'))

    for vector, label in data_loader:
        vector, label = vector.to(device), label.to(device)
        logit = F.softmax(net(vector), dim=-1)
        pred = logit.argmax(-1)
        for p, l in zip(pred, label):
            preds.append(int(p))
            labels.append(int(l))

    return accuracy_score(labels, preds)


if __name__ == '__main__':
    args = create_parser().parse_args()
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataset = torch.load('data/train_dataset.pt')
    train_data_loader = DataLoader(train_dataset, batch_size=250, shuffle=True)
    dev_dataset = torch.load('data/dev_dataset.pt')
    dev_data_loader = DataLoader(dev_dataset, batch_size=50, shuffle=False)

    time = timeit.timeit('train(train_data_loader, 100, device)', globals=globals(), number=1)
    # train(train_data_loader, 100, device)
    acc = calc_accuracy(dev_data_loader, device)
    print(f'time:{time}, accuracy:{acc}')
    with open('data/result.txt','w') as f:
        f.write(f'time:{time}, accuracy:{acc}\n')

    # 1epochあたりの所用時間を計算
    # file = open('data/time_count.txt','w')
    # batch_sizes = [2**i for i in range(15)]
    # print(batch_sizes)
    # for batch_size in batch_sizes:
    #     train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     result = timeit.timeit('train(train_data_loader, 1)', globals=globals(), number=1)
    #     print(f'batch_size {batch_size}: {result}')
    #     file.write(f'batch_size {batch_size}: {result}\n')