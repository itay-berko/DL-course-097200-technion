import json
import sys

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import datetime

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


class Config:
    def __init__(self, batch_size, num_epochs, lr, dropout, schedular_step_size):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.dropout = dropout
        self.schedular_step_size = schedular_step_size

def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def get_data_loaders(config):
    # Image Preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2434, 0.2615)),
    ])

    # CIFAR-10 Dataset
    train_dataset = dsets.CIFAR10(root='./data/',
                                  train=True,
                                  transform=transform_train,
                                  download=True)

    test_dataset = dsets.CIFAR10(root='./data/',
                                 train=False,
                                 transform=transform_test,
                                 download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)
    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.dropout1 = nn.Dropout(p=config.dropout)
        self.fc1 = nn.Linear(4 * 4 * 32, 40)
        self.batch1 = nn.BatchNorm1d(40)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(40, 10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)

        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.batch1(out)
        out = self.relu1(out)

        out = self.fc2(out)

        return self.logsoftmax(out)


def get_cnn(config):
    cnn = CNN(config)
    cnn = to_gpu(cnn)

    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
    print('Num of trainable parameters : %'.format(sum(p.numel() for p in cnn.parameters() if p.requires_grad)))
    return cnn


# Test the Model
def test_model(curr_model, test_loader, criterion):
    curr_model.eval()
    curr_model = curr_model.cpu()
    correct = 0
    total = 0
    epoch_loss = 0
    for images, labels in test_loader:
        # images = images.view(-1, 28*28)
        # images = to_gpu(images)
        # labels = to_gpu(labels)
        outputs = curr_model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss = criterion(outputs, labels)
        epoch_loss += loss * len(labels) / len(test_loader.dataset)
    curr_model_test_acc = float(correct) / float(total)
    return curr_model_test_acc, epoch_loss


def train_model(config):
    # Training the Model
    max_acc = 0
    train_acc_vec = []
    test_acc_vec = []
    train_loss_vec = []
    test_loss_vec = []
    net = get_cnn(config)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=config.schedular_step_size, gamma=0.9)
    train_loader, test_loader = get_data_loaders(config)
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    progress_file = start_time_str + '_progress.csv'
    with open(start_time_str + '_config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print(str(datetime.datetime.now()) + ' training started')

    for epoch in range(config.num_epochs):
        # print(str(datetime.datetime.now()) + ' train')
        net.train()
        net = to_gpu(net)
        epoch_acc = 0
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # images = images.view(-1, 28*28).to()
            images = to_gpu(images)
            labels = to_gpu(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            acc_all = (outputs.max(dim=1).indices == labels)
            acc = float(acc_all.sum()) / len(labels)
            epoch_acc += acc * len(labels) / len(train_loader.dataset)
            epoch_loss += loss * len(labels) / len(train_loader.dataset)
            loss.backward()
            optimizer.step()

        # print(str(datetime.datetime.now()) + ' test')
        test_acc, test_loss = test_model(net, test_loader, criterion)
        scheduler.step()

        test_acc_vec.append(float(test_acc))
        train_acc_vec.append(float(epoch_acc))
        test_loss_vec.append(float(test_loss))
        train_loss_vec.append(float(epoch_loss))
        print(str(datetime.datetime.now()) +
              ' Epoch: [%d/%d], lr: %.6f, train loss: %.4f, test loss: %.4f, train acc: %.4f, test acc: %4f'
            %(epoch+1, config.num_epochs, optimizer.param_groups[0]['lr'], epoch_loss, test_loss, epoch_acc, test_acc))
        write_progress(train_acc_vec, train_loss_vec, test_acc_vec, test_loss_vec, progress_file)
        if test_acc > max_acc and test_acc > 0.8:
            print('model saved')
            torch.save(net.state_dict(), start_time_str + '_best_model.pkl')
            max_acc = test_acc
    print(str(datetime.datetime.now()) + ' training complete. Maximal test accuracy: %4f'%(max_acc))
    # display_progress(progress_file)


def write_progress(train_acc, train_loss, test_acc, test_loss, filename):
    with open(filename, 'w') as f:
        f.write('train_acc,train loss,test acc,test loss\n')
        for i in range(len(train_acc)):
            f.write('%.4f,%.4f,%.4f,%.4f\n'%(train_acc[i], train_loss[i], test_acc[i], test_loss[i]))


def display_progress(filename):
    progress_dict = json.load(open(filename))
    num_epochs = len(progress_dict['train_acc'])
    plt.subplot(2, 1, 1)
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(range(num_epochs), progress_dict['train_acc'], 'r', label='Train')
    plt.plot(range(num_epochs), progress_dict['test_acc'], 'b', label='Test')
    plt.legend(loc="lower right")

    plt.subplot(2, 1, 2)
    plt.title('Loss per epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(range(num_epochs), progress_dict['train_loss'], 'r', label='Train')
    plt.plot(range(num_epochs), progress_dict['test_loss'], 'b', label='Test')
    plt.legend(loc="upper right")

    plt.subplots_adjust(hspace=1)
    plt.show()


def run():
    # Hyper Parameters
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    start_config = Config(batch_size=batch_size,
                          num_epochs=num_epochs,
                          lr=0.001,
                          dropout=0.1,
                          schedular_step_size=40
                          )
    train_model(start_config)


if __name__ == "__main__":
    run()
