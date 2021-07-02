from torchvision import datasets
# from torchtext.datasets import WikiText2

cifar = datasets.CIFAR10(root='/home/ubuntu/measurement/data', train=True, download=True)
# mnist = datasets.MNIST(root='/home/ubuntu/measurement/data', train=True, download=True)
# wiki = WikiText2(root='../../data', split=('train', 'valid', 'test'))

