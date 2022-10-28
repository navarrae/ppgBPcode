import torch
import torch.nn as nn
import pandas as pd                     # to process our data
import matplotlib.pyplot as plt         # graphing
# from utils import decision_boundary     # for plotting
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from sklearn.metrics import mean_squared_error

 

torch.manual_seed(1)

ppg_data = pd.read_csv('ppg_train.csv')
num_data_points = ppg_data.shape
print("There are", num_data_points, "data points.")


#ppg = torch.tensor(ppg_data[['max1', 'min1', 'max2','min2', 'max3', 'min3']].values,dtype=torch.float32)
#bp = torch.tensor(ppg_data['sbp'].values,dtype = torch.float32)


ppg = torch.tensor(ppg_data[['max1', 'min1', 'max2','min2', 'max3', 'min3']].values,dtype=torch.float32)
bp = torch.tensor(ppg_data['dbp'].values,dtype = torch.float32)

#print(ppg)
#print(bp)

class DeepNeuralNetwork(torch.nn.Module):
     def __init__(self):
        super(DeepNeuralNetwork, self).__init__()
        self.lin1 = torch.nn.Linear(6,20)
        self.lin2 = torch.nn.Linear(20,1)
     def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        #x = torch.relu(x)
        #x = self.lin3(x)
        return x

dl_model = DeepNeuralNetwork()
print(dl_model.parameters())
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(dl_model.parameters(), lr = 0.001, weight_decay=0.001) 

loss_train = []


for epoch in range(500000): 
  # TODO: fill in the blanks with your own code! 
    # resets the information from last time
    optimizer.zero_grad()
    # calculates the predictions
    pred_y = dl_model(ppg)
    #print(pred_y)
    pred_y = pred_y.flatten()
    # calculates the loss
    loss = criterion(pred_y, bp)
    # gradient descent, part 1
    loss.backward()
    # gradient descent, part 2
    optimizer.step()
    # print loss every 500 epochs
    if epoch % 5000 == 0:
        #print(pred_y)
        print(f"{epoch}:{loss.item()}")

torch.save(dl_model.state_dict(), 'state_dict')




