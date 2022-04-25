import pandas as pd
import numpy as np
import torch
import math

class myDataLoader():
    def __init__(self):
        dtype=torch.float

        #load our data
        #test file is stock prices
        #edit this code for user to specify data type, e.g. Covid cases, temperature, prices, etc.
        data = pd.read_csv("data/datafile.csv")
        open = data['Entries']

        self.data_tensor = torch.tensor(open.values,dtype=dtype)

        num_ts_in=1 #you should have it figure this out on its own
        self.data_tensor=self.data_tensor.view(-1,num_ts_in)

        print(f"self.data_tensor.size()={self.data_tensor.size()}")

    def get_data(self):
        return self.data_tensor
