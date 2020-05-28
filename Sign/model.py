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
      self.final_pos = None
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
        path = os.path.join('/content/drive/My Drive/Colab Notebooks/DL', 'model'+'.pkl')
        self.classifier.load_state_dict(torch.load(path))
      self.output = self.classifier(self.input)
      self.loss = self.loss_function(self.output, self.label)
    
    # ROA_Algorithm_1: ExhaustiveSearching
    def roa_exhasted(self, N, W, H, S, pretrained = False):
      self.classifier = self.classifier.eval()
      if pretrained:
        path = os.path.join('/content/drive/My Drive/Colab Notebooks/DL', 'model'+'.pkl')
        self.classifier.load_state_dict(torch.load(path))
      
      min_ = 1e4
      rectangle = 0.5 * torch.ones((3,H,W))
      for i in range((N-H)//S):
          for j in range((N-W)//S):
    
            image = self.input.clone()
            image[:,:,S*i:S*i+H, S*j:S*j+W] = rectangle
            output = self.classifier(image.to(self.device))
            index = output.argmax(dim=1)
            index = index.cpu().detach()
            target = output[:,index].item()

            if target < min_:
              min_ = target
              self.final_pos = (i,j)
                
    # Helper Searching
    def helpersearching(self, grad, N, width, height, S, C):
      iterx = (N-width) // S
      itery = (N-height) // S
      losslist = []
      dict_ = {}
      for j in range(iterx):
          for k in range(itery):
            g = grad[:,:, S*j: S*j+height, S*k: S*k+width]
            loss = torch.sum(torch.sum(torch.sum(torch.mul(g,g),1),1),1)
            dict_[(j, k)] = loss
      self.dict_pos = sorted(dict_.items(),key=lambda item:item[1],reverse=True)[:C]
    
    # ROA_Algorithm_2: GradientBasedSearch
    def gradient_based_search(self, N, width, height, S, C, pretrained = False):
      self.classifier = self.classifier.eval()

      self.input = self.input.requires_grad_(True)
      if pretrained:
        path = os.path.join('/content/drive/My Drive/Colab Notebooks/DL', 'model'+'.pkl')
        self.classifier.load_state_dict(torch.load(path))

      self.output = self.classifier(self.input)
      self.loss = self.loss_function(self.output, self.label)
      self.loss.sum().backward()
      grad = self.input.grad.data

      max_val,_ = torch.max(torch.abs(grad.view(grad.shape[0], -1)),1)
      grad = grad/max_val

      self.helpersearching(grad, N, width, height, S, C)

      max_loss = -1
      for pos, _ in self.dict_pos:
          rectangle = 0.5 * torch.ones((3,height,width))
          attack_image = self.input.clone()
          attack_image[:,:, S*pos[0]:S*pos[0]+height, S*pos[1]:S*pos[1]+width] = rectangle
          self.output = self.classifier(attack_image)
          X, pred = torch.max(self.output.detach(), dim=1)
          self.loss = self.loss_function(self.output, self.label)
          if max_loss <= self.loss.data:
            max_loss = self.loss.data
            self.final_pos = pos
      
      # attack_image[:,:, S*self.final_pos[0]:S*self.final_pos[0]+height, S*self.final_pos[1]:S*self.final_pos[1]+width] = rectangle
      # self.output = self.classifier(attack_image)
      # _, pred = torch.max(self.output.detach(), dim=1)
      # print(pred, self.label)

      # im = attack_image.squeeze(0).cpu().detach().numpy()*255
      # r = Image.fromarray(im[0]).convert('L')
      # g = Image.fromarray(im[1]).convert('L')
      # b = Image.fromarray(im[2]).convert('L')
      # ima = Image.merge("RGB", (r, g, b))
      # plt.imshow(ima)
      
    def PGD(self, N, W, H, S, iter, lr, initial = 'half'):
        mask = torch.zeros((N, N), requires_grad=True)
        mask[S*self.final_pos[0]:S*self.final_pos[0]+H, 
             S*self.final_pos[1]:S*self.final_pos[1] + W] = 1.0
        mask = mask.to(self.device)

        # Initialize the input image before PGD
        if initial == 'random':
          patch = torch.rand_like(self.input, requires_grad=True).to(self.device)
        elif initial == 'half':
          patch = (torch.zeros_like(self.input, requires_grad=True) + 1/2).to(self.device)
        
        image_adv = torch.rand_like(self.input, requires_grad=True).to(self.device)
        image_adv.data = self.input * (1-mask) + patch*mask

        # PGD interation
        for epoch in range(iter):
            self.output = self.classifier(image_adv)
            loss = self.loss_function(self.output, self.label)
            loss.backward()
            image_adv.data = image_adv.detach() + lr * image_adv.grad.detach().sign() * mask

            image_adv.data = (image_adv.detach()).clamp(0,255)
            image_adv.grad.zero_()


        # self.output = self.classifier(image_adv)
        # _, pred = torch.max(self.output.detach(), dim=1)
        # print(pred, self.label)

#         im = image_adv.squeeze(0).cpu().detach().numpy()*255
#         r = Image.fromarray(im[0]).convert('L')
#         g = Image.fromarray(im[1]).convert('L')
#         b = Image.fromarray(im[2]).convert('L')
#         ima = Image.merge("RGB", (r, g, b))
#         plt.imshow(ima)

 
        return image_adv.detach()

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
