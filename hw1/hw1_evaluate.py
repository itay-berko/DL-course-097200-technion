import torch
import torchvision
import torch.nn as nn


def Net(input_size, num_classes):
    drop = 0.4
    n_h = 75
    n_h2 = 32
    return nn.Sequential(
        nn.BatchNorm1d(input_size),
        nn.Linear(input_size, n_h),
        nn.ReLU(),
        nn.BatchNorm1d(n_h),
        nn.Dropout(drop),
        nn.Linear(n_h, n_h2),
        nn.ReLU(),
        nn.BatchNorm1d(n_h2),
        nn.Dropout(drop),
        nn.Linear(n_h2, num_classes),
    )


def evaluate_hw1():
    batch_size_test = 128

    use_gpu = torch.cuda.is_available()

    def to_cuda(x):
        if use_gpu:
            x = x.cuda()
        return x

    train_dataset = torchvision.datasets.FashionMNIST('/', train=False, download=True,
                                                      transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_test, shuffle=True)

    input_size = 784
    num_classes = 10

    network = Net(input_size,num_classes)
    network.load_state_dict(torch.load('net.pkl'))
    network = to_cuda(network)
    criterion = nn.CrossEntropyLoss()

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = to_cuda(data.view(-1, 28 * 28))
                target = to_cuda(target)
                outputs = network(data)
                test_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target).sum()

        test_loss /= len(test_loader.dataset)
        accuracy = float(correct) / len(test_loader.dataset)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return accuracy

    acc = test()

    return acc


if __name__ == "__main__":
    my_acc = evaluate_hw1()
    print(my_acc)
