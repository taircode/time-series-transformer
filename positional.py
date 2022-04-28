import torch
from torch import Tensor
import torch.nn as nn
import math

#Originally this was not subclassing nn.Module
class myPositionalEncoding(nn.Module):
    def __init__(self, pe_features: int, seq_length: int, pe_type: str='add'):
        #super().__init__() #need this if subclassing nn.Module

        self.pe_type=pe_type

        batch_size=1

        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_features, 2) * (-math.log(10000.0) / pe_features))
        pe = torch.zeros(seq_length,pe_features)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #pe = pe.unsqueeze(0)

        #There's a small difference between doing self.pe with require_grad=False and registering a buffer
        #buffers get saved as state_dict and get moved to cuda when the model is, but don't have SGD applied to them
        #https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch
        #https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        #self.pe.require_grad = False
        self.register_buffer('pe',pe)

    def forward(self, x, type):
        if self.pe_type=='add':
            output=torch.add(x,self.pe)
            return output
        else: #type=='concat'
            output=torch.cat((x,self.pe),1)
            return output

    """
    These are old functions. Before this class subclassed nn.Module
    """
    #seq_length, num_pe_dimensions #note no batches right now
    def concat(self, x: Tensor) -> Tensor:
        #print(f"self.pe.size()={self.pe.size()}")
        #print(f"x.size()={x.size()}")
        output = torch.cat((x,self.pe),1)
        return output

    def add(self, x: Tensor) -> Tensor:
        output=torch.add(x,self.pe)
        return output