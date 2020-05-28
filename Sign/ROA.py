import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from model import Model 

path = './LISA'

transform = transforms.Compose([
           transforms.Resize(size = (32,32)),
           transforms.ToTensor(),
           ])

trainset = ImageFolder(os.path.join(path, 'train'), transform)
print('trainset: %d' % len(trainset))
valset = ImageFolder(os.path.join(path, 'val'), transform)
print('valset: %d' % len(valset))
testset = ImageFolder(os.path.join(path, 'test'), transform)
print('testset: %d' % len(testset))

trainloader = DataLoader(trainset, batch_size = 128, shuffle = True)
valloader = DataLoader(valset, batch_size = 128, shuffle = False)
testloader = DataLoader(testset, batch_size = 1, shuffle = False)

model = Model()

accuracy, loss = 0, 0
for i, data in enumerate(trainloader):
    input, label = data
    dir = '/content/drive/My Drive/Colab Notebooks/DL/ROA_LISA/train/%d' % label
    model.set_input(input, label)
    model.roa_exhasted(N=32, W=7, H=7, S=2, pretrained=True)
    # model.roa_exhasted(N=32, W=10, H=5, S=2, pretrained=True)
    # model.gradient_based_search(N=32, width=7, height=7, S=2, C=10, pretrained=True)
    # model.gradient_based_search(N=32, width=10, height=5, S=2, C=10, pretrained=True)
    X = model.PGD(N=32, W=7, H=7, S=2, iter=50, lr=4, initial='half')
    # X = model.PGD(N=32, W=10, H=5, S=2, iter=50, lr=4, initial='half')
    
    X = X.squeeze(0).cpu().detach().numpy()*255
    mkdir(dir)
    r = Image.fromarray(X[0]).convert('L')
    g = Image.fromarray(X[1]).convert('L')
    b = Image.fromarray(X[2]).convert('L')
    ima = Image.merge("RGB", (r, g, b))
    ima.save(dir + '/%d.png'%i)

for i, data in enumerate(valloader):
    input, label = data
    dir = '/content/drive/My Drive/Colab Notebooks/DL/ROA_LISA/val/%d' % label
    model.set_input(input, label)
    model.roa_exhasted(N=32, W=7, H=7, S=2, pretrained=True)
    # model.roa_exhasted(N=32, W=10, H=5, S=2, pretrained=True)
    # model.gradient_based_search(N=32, width=7, height=7, S=2, C=10, pretrained=True)
    # model.gradient_based_search(N=32, width=10, height=5, S=2, C=10, pretrained=True)
    X = model.PGD(N=32, W=7, H=7, S=2, iter=50, lr=4, initial='half')
    # X = model.PGD(N=32, W=10, H=5, S=2, iter=50, lr=4, initial='half')
    
    X = X.squeeze(0).cpu().detach().numpy()*255
    mkdir(dir)
    r = Image.fromarray(X[0]).convert('L')
    g = Image.fromarray(X[1]).convert('L')
    b = Image.fromarray(X[2]).convert('L')
    ima = Image.merge("RGB", (r, g, b))
    ima.save(dir + '/%d.png'%i)

for i, data in enumerate(testloader):
    input, label = data
    dir = '/content/drive/My Drive/Colab Notebooks/DL/ROA_LISA/test/%d' % label
    model.set_input(input, label)
    model.roa_exhasted(N=32, W=7, H=7, S=2, pretrained=True)
    # model.roa_exhasted(N=32, W=10, H=5, S=2, pretrained=True)
    # model.gradient_based_search(N=32, width=7, height=7, S=2, C=10, pretrained=True)
    # model.gradient_based_search(N=32, width=10, height=5, S=2, C=10, pretrained=True)
    X = model.PGD(N=32, W=7, H=7, S=2, iter=50, lr=4, initial='half')
    # X = model.PGD(N=32, W=10, H=5, S=2, iter=50, lr=4, initial='half')
    
    X = X.squeeze(0).cpu().detach().numpy()*255
    mkdir(dir)
    r = Image.fromarray(X[0]).convert('L')
    g = Image.fromarray(X[1]).convert('L')
    b = Image.fromarray(X[2]).convert('L')
    ima = Image.merge("RGB", (r, g, b))
    ima.save(dir + '/%d.png'%i)
