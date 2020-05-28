import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, 16)
        

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        output = self.fc(x)

        return output

class Model():
    def __init__(self):
      self.device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
      self.classifier = Network().to(self.device)
      self.loss_function = nn.CrossEntropyLoss()
      self.lr = 0.03
      self.optim = optim.SGD(self.classifier.parameters(), lr=0.03, momentum=0.9)
    
    def set_input(self, input, label):
      self.input, self.label = input.to(self.device), label.to(self.device)
    
    def optimizer(self):
      self.output = self.classifier(self.input)
      self.optim.zero_grad()
      self.loss = self.loss_function(self.output, self.label)
      self.loss.backward()
      self.optim.step()
    
    def test(self, pretrained = False):
      self.classifier = self.classifier.eval()
      if pretrained:
        path = os.path.join('./', 'model'+'.pkl')
        self.classifier.load_state_dict(torch.load(path))
      self.output = self.classifier(self.input)
      self.loss = self.loss_function(self.output, self.label)

    def get_current_error(self):
      _, pred = torch.max(self.output.detach(), dim=1)
      correct_mask = torch.eq(pred, self.label).type(torch.FloatTensor)
      self.accuracy = torch.mean(correct_mask)
      
      return self.accuracy.item(), self.loss.item()
  
    def print(self, epoch, i, iter_):
      print('Epoch: %d(%d/%d) current_accuracy: %.3f current_Train_loss: %.3f' % (epoch, i, iter_, self.accuracy, self.loss))

    def update_learning_rate(self, ratio):
        lr = self.lr * ratio
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        print('update classifier learning rate: %f -> %f' % (self.lr, lr))
        self.lr = lr

    def save_net(self):
      save_path = os.path.join('./', 'model'+'.pkl')
      torch.save(self.classifier.state_dict(), save_path)
