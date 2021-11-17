import argparse
import pickle
import sys
import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score


'''f
python train_Q89.py -device 0 -me 50 -bs 128
rsync -avzhP amaretto01:/home/SeiyaKikuchi/100knock-2020/SeiyaKikuchi/chapter09/data/loss_q89.png /Users/seiya.k/workspace/100knock-2020/SeiyaKikuchi/chapter09/data/
rsync -avzhP amaretto01:/home/SeiyaKikuchi/100knock-2020/SeiyaKikuchi/chapter09/data/accuracy_q89.png /Users/seiya.k/workspace/100knock-2020/SeiyaKikuchi/chapter09/data/
rsync -avzhP /Users/seiya.k/workspace/100knock-2020/SeiyaKikuchi/chapter09/data/weight.pt amaretto01:/home/SeiyaKikuchi/100knock-2020/SeiyaKikuchi/chapter09/data/
'''


class MyModel(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.transformer = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        self.fc = nn.Linear(768, output_size)

    def forward(self, idseqs):
        out = self.transformer(idseqs)
        hidden = out['last_hidden_state']
        logit = out['pooler_output']
        logit = self.fc(logit)
        return logit


def test(model, device, data_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        y_true, y_pred = [], []

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_func(out, y)

            total_loss += loss
            pred = out.argmax(-1)
            for p, t in zip(pred, y):
                y_true.append(int(t))
                y_pred.append(int(p))
        loss_ave = total_loss / x.shape[0]
        print(loss_ave)
        acc = accuracy_score(y_true, y_pred)
        print(acc)

        return loss_ave, acc


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


def draw_graph(process, train_result, test_result):
    #lossの推移
    sns.set()
    sns.set_palette('viridis')
    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('epoch')
    plt.ylabel(process)
    plt.plot(range(args.max_epoch), train_result, label='train')
    plt.plot(range(args.max_epoch), test_result, label='test')
    plt.legend()
    fig.savefig(f"{process}.png")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", dest="device_id", type=int,
                        help="Serect device id on remote server", default=0)
    parser.add_argument("-bs", dest="batch_size", type=int,
                        help="decide batch size", default=32)
    parser.add_argument("-me", dest="max_epoch", type=int,
                        help="decide batch size", default=10)
    parser.add_argument("-p", dest="path", type=str,
                        help="input path to directory for experience", default='data_folder')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataset, test_dataset = torch.load('../data/train.pt'), torch.load('../data/test.pt')
    train_data_loader, test_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
                                          DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    word2id = pickle_load('../data/word2id.pickle')
    vocab_size = len(word2id)

    model = MyModel(output_size=4)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()

    losses, accs, test_losses, test_accs = [], [], [], []

    for epoch in range(args.max_epoch):
        model.train()
        y_true, y_pred = [], []
        total_loss = 0

        for x, y in tqdm(train_data_loader, desc="train", position=0, file=sys.stdout):
            x, y = x.to(device), y.to(device)
            model = model.to(device)
            out = model(x)
            optimizer.zero_grad()
            loss = loss_func(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss
            pred = out.argmax(-1)
            for p, t in zip(pred, y):
                y_pred.append(int(p))
                y_true.append(int(t))

        loss_ave = total_loss / x.shape[0]
        acc = accuracy_score(y_true, y_pred)
        print('train: ', loss_ave)
        print('train: ', acc)
        losses.append(loss_ave)
        accs.append(acc)

        #評価
        test_loss, test_acc = test(model, device, test_data_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    draw_graph('../data/loss_q89', losses, test_losses)
    draw_graph('../data/accuracy_q89', accs, test_accs)