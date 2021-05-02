import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim):
        
        super(SimpleNet,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,output_dim)
        self.drop=nn.Dropout(0.3)
        self.sig=nn.Sigmoid()
        
    def forward(self,x):
        
        out=F.relu(self.fc1(x))
        out=self.drop(out)
        out=self.fc2(out)
        return self.sig(out)