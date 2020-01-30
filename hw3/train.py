import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
    epoch_acc = 0
    epoch_loss = 0

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax(dim=1).cuda()
    for v, q, a, idx, q_len in tq:
        var_params = {
            'requires_grad': False,
        }
        v = Variable(v.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)

        out = net(v, q, q_len)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()
        epoch_acc += float(acc.float().mean()) * len(q_len) / len(loader.dataset)
        epoch_loss += float(loss * len(q_len) / len(loader.dataset))

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iterations += 1
        
        loss_tracker.append(loss.data.item())
        # acc_tracker.append(acc.mean())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    return epoch_loss, epoch_acc


def main():
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    progress_file = start_time_str + '_progress.csv'

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    max_acc = 0

    for i in range(config.epochs):
        print('epoch %d' % i)
        print('train')
        train_loss, train_acc = run(net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i)
        print('validation')
        val_loss, val_acc = run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        write_progress(train_acc_list, train_loss_list, val_acc_list, val_loss_list, progress_file)
        if val_acc > max_acc and val_acc > 0.45:
            print('model saved')
            torch.save(net.state_dict(), start_time_str + '_best_model.pkl')
            max_acc = val_acc

def write_progress(train_acc, train_loss, test_acc, test_loss, filename):
    with open(filename, 'w') as f:
        f.write('train_acc,train loss,test acc,test loss\n')
        for i in range(len(train_acc)):
            f.write('%.4f,%.4f,%.4f,%.4f\n'%(train_acc[i], train_loss[i], test_acc[i], test_loss[i]))


def display_progress(filename):
    progress_dict = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    progress_f = open(filename)
    for row_idx, row in enumerate(progress_f):
        if row_idx == 0:
            continue
        train_acc, train_loss, test_acc, test_loss = row.rstrip().split(',')
        progress_dict['train_acc'].append(float(train_acc))
        progress_dict['train_loss'].append(float(train_loss))
        progress_dict['test_acc'].append(float(test_acc))
        progress_dict['test_loss'].append(float(test_loss))

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


if __name__ == '__main__':
    main()
    # display_progress('2020-01-28_00-25-42_progress.csv')