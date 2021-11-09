import pandas as pd
import numpy as np
import torch
import math

class myDataLoader():
    def __init__(self):
        dtype=torch.float

        #load our data
        btc_daily = pd.read_csv("data/btc_daily_raw.csv")
        open = btc_daily['Open']

        self.data_tensor = torch.tensor(open.values,dtype=dtype)

        num_ts_in=1 #you should have it figure this out on its own
        self.data_tensor=self.data_tensor.view(-1,num_ts_in)

        print(f"self.data_tensor.size()={self.data_tensor.size()}")

    def get_data(self):
        return self.data_tensor
