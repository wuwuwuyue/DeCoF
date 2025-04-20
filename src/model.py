import torch
from torch import nn
from utils.sam import SAM
from vit import ViT
from clip_models import  CLIPModel


class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net=CLIPModel(name='ViT-L/14')
        for name, p in self.net.named_parameters():
            p.requires_grad = False

        self.net_all=ViT()   
        self.cel=nn.CrossEntropyLoss()
        self.optimizer=SAM(self.net_all.parameters(),torch.optim.SGD,lr=0.001,momentum=0.9)
    def forward(self,x):
        y=torch.zeros(len(x),8,768).cuda()
        
        for i in range(len(x)):
           
            img=x[i]
            img=self.net(img)
          
            y[i]=img
        x=self.net_all(y)
        return x 
        
    def training_step(self,x,target):
        for i in range(2):
            pred_cls=self(x)
            #print(x.shape)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            loss=loss_cls
            self.optimizer.zero_grad()
            loss.backward()
           
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
           
        return pred_first
    
if __name__=='__main__':
    model=Detector()
        
    img = torch.randn(16,8,3,224, 224)

    preds = model(img) 

    print(preds.shape)
