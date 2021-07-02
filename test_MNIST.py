import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging
from model import cnn


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            logging.warning('Epoch: {:2d} | Batch: {:3d}/{:4d} | loss: {:.2f}'.format(epoch+1, batch_idx, len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.warning('Test loss: {:.2f} | Accuracy: {:.2f}%'.format(test_loss, 100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str, default="../data", help="The location of dataset.")
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='batch size for testing')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename='/workspace/measurement/logs/test_MNIST.logs', filemode='w')
    logging.warning('Data downloading')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = cnn.CNN().to(device)
    for p in model.parameters():
        p.grad
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
