import torch
from torch import Tensor
import torch.nn as nn
import math

class myPositionalEncoding():
    def __init__(self, pe_features: int, seq_length: int):
        super().__init__()

        batch_size=1

        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_features, 2) * (-math.log(10000.0) / pe_features))
        self.pe = torch.zeros(seq_length,pe_features)
        print(f"self.pe.size()={self.pe.size()}")
        print(f"pe_features={pe_features}")
        
        #note this is for batch_size=1
        self.pe[:, 0::2] = torch.sin(position * div_term)

        self.pe[:, 1::2] = torch.cos(position * div_term)

    #seq_length, num_pe_dimensions #note no batches right now
    def concat(self, x: Tensor) -> Tensor:
        output = torch.cat((x,self.pe),1)
        return output

    def add(self, x: Tensor) -> Tensor:
        output=torch.add(x,self.pe)
        return output