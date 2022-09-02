# -*- coding: UTF-8 -*-
import torch.nn as nn
import captcha_setting

import torch
import torchvision.models as models

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((captcha_setting.IMAGE_WIDTH//8)*(captcha_setting.IMAGE_HEIGHT//8)*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = nn.Sequential()
        
        num = 1.8
        
        for name,i in models.resnet18().named_children():
            
            if 'layer' in name:
                self.resnet.add_module(name, i)
                self.resnet.add_module(name+'dropout',nn.Dropout(1/num))
                num = num +1
            else:
                self.resnet.add_module(name, i)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512,captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
        )
        
        for m in self.resnet.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                
        return None
    
    def forward(self, x):
        
        return self.resnet(x)

#%%
if __name__ == '__main__':
    resnet18 = models.resnet18()
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet18.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512,captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
        )
    
    resnet18 = resnet()
    
    X = torch.rand((1,1,160,60))
    mid = 0

    for name,i in resnet18.resnet.named_children():
    
        print(name)
        X = i(X)
        print(X.shape) 

    
    