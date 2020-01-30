import torch
from torch import nn, optim

import config
from image_manager import preprocess_images
import data
import model
from text_manager import preprocess_text
from train import run
import utils


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def evaluate_hw3():
    print('preprocessing images')
    preprocess_images(train=False)
    print('preprocessing text')
    preprocess_text()
    print('loading data')
    val_loader = data.get_loader(val=True)
    net = nn.DataParallel(model.Net(config.default_num_tokens)).cuda()
    net.load_state_dict(torch.load('model.pkl', map_location=lambda storage, loc: storage))
    net = to_gpu(net)
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    tracker = utils.Tracker()
    print('runnning validation')
    val_loss, val_acc = run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=0)
    print('validation acc: %.4f' % val_acc)


if __name__ == '__main__':
    evaluate_hw3()