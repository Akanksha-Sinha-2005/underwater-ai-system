import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()

        self.encoder1=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU()
        )
        self.pool=nn.MaxPool2d(2,2)

        self.encoder2=nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU()
        )
        self.up=nn.ConvTranspose2d(128,64,2,stride=2)

        self.decoder=nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,3,3,padding=1),
            nn.Sigmoid()
        )

    
    def forward(self,x):
        e1=self.encoder1(x)
        p1=self.pool(e1)
        e2=self.encoder2(p1)
        u=self.up(e2)
        if u.shape!=e1.shape:
            u=F.interpolate(u,size=e1.shape[2:])
        cat=torch.cat([u,e1],dim=1)
        out=self.decoder(cat)
        return out
    
