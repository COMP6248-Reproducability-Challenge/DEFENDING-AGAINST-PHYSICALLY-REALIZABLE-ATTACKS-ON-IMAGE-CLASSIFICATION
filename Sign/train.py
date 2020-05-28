import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
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
testloader = DataLoader(testset, batch_size = 128, shuffle = False)

model = Model()

for epoch in range(1, 31):
    accuracy, loss = 0, 0
    for i, data in enumerate(trainloader):
      input, label = data
      model.set_input(input, label)
      model.optimizer()
      temp_accuracy, temp_loss = model.get_current_error()
      accuracy, loss = accuracy+temp_accuracy, loss+temp_loss
      if i%5 == 0:
        model.print(epoch, i, len(trainloader))
    print('Epoch: %d Average_Accuracy: %.3f Average_Train_loss: %.3f' % (epoch, 
                                                        (accuracy/len(trainloader)), 
                                                        (loss/len(trainloader))))  
    model.save_net()
    print('Saving network...') 
    if epoch%10 == 0:
      model.update_learning_rate(0.1)

#val
accuracy, loss = 0, 0
for _, data in enumerate(valloader):
  input, label = data
  model.set_input(input, label)
  model.test(pretrained = True)
  temp_accuracy, temp_loss = model.get_current_error()
  accuracy, loss = accuracy+temp_accuracy, loss+temp_loss
print('val_Accuracy: %.3f val_loss: %.3f' % ( accuracy/len(valloader), 
                                                      loss/len(valloader)))

#test
accuracy, loss = 0, 0
for i, data in enumerate(testloader):
  input, label = data
  model.set_input(input, label)
  model.test(pretrained = True)
  temp_accuracy, temp_loss = model.get_current_error()
  accuracy, loss = accuracy+temp_accuracy, loss+temp_loss

print('test_Accuracy: %.3f Test_loss: %.3f' % (accuracy/len(testloader), 
                                          loss/len(testloader)))
