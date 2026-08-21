import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
from torch.nn.modules.activation import ReLU
import torchvision
from torchvision import transforms


train_transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32,padding=4),
                                    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2470, 0.2435, 0.2616))])

validation_transform=transforms.Compose(transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2470, 0.2435, 0.2616)))    

train_dataset= torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=train_transform)
validation_dataset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=validation_transform)


train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=2)
validation_loader=DataLoader(validation_dataset,batch_size=64,shuffle=False,num_workers=2)



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.feature=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,192,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(192),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(192,384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLu(),
        

            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLu(),
            

            nn.Conv2d(256,256,kernel_size=3,stride=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

        )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x=self.feature(x)   
        x=self.classification(x)
        return x   

            






        








                                         
                                                                        