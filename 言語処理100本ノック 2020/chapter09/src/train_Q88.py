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


'''
python train_Q88.py -device 1 -me 100 -bs 64
rsync -avzhP amaretto01:/home/SeiyaKikuchi/100knock-2020/SeiyaKikuchi/chapter09/data/loss_q88.png /Users/seiya.k/workspace/100knock-2020/SeiyaKikuchi/chapter09/data/
rsync -avzhP amaretto01:/home/SeiyaKikuchi/100knock-2020/SeiyaKikuchi/chapter09/data/accuracy_q88.png /Users/seiya.k/workspace/100knock-2020/SeiyaKikuchi/chapter09/data/
rsync -avzhP /Users/seiya.k/workspace/100knock-2020/SeiyaKikuchi/chapter09/data/weight.pt amaretto01:/home/SeiyaKikuchi/100knock-2020/SeiyaKikuchi/chapter09/data/
'''


class MyRnn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, padding_id):
        super(MyRnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_id)
        self.rnn = nn.RNN(embed_dim, hidden_dim, bias=True, batch_first=True,
                          bidirectional=True, num_layers=4)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x, device):
        batch_size = x.shape[0]
        length = [torch.count_nonzero(idseq.detach() != 0) for idseq in x]
        hidden = torch.zeros(8, batch_size, self.hidden_dim)
        hidden = hidden.to(device)
        embeds = self.embedding(x)
        packed_input = pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn(packed_input, hidden)
        logit = self.fc(hidden)

        return logit, hidden


def test(model, device, data_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        y_true, y_pred = [], []

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out, hidden = model(x, device)
            loss = loss_func(out[0], y)

            total_loss += loss
            pred = out[0].argmax(-1)
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
    fig.savefig(f"{process}_q88.png")


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

    model = MyRnn(vocab_size, 300, 50, 4, padding_id=0)
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
            out, hidden = model(x, device)
            optimizer.zero_grad()
            loss = loss_func(out[0], y)
            loss.backward()
            optimizer.step()

            total_loss += loss
            pred = out[0].argmax(-1)
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

    draw_graph('../data/loss', losses, test_losses)
    draw_graph('../data/accuracy', accs, test_accs)