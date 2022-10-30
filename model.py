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

valid = pd.read_csv('ppg_valid.csv')
ppg_val = torch.tensor(valid[['max1', 'min1', 'max2','min2', 'max3', 'min3']].values,dtype=torch.float32)

expected = torch.tensor(valid['sbp'].values,dtype = torch.float32)

#print(ppg)
#print(bp)

class DeepNeuralNetwork(torch.nn.Module):
     def __init__(self):
        super(DeepNeuralNetwork, self).__init__()
        self.lin1 = torch.nn.Linear(6,100)
        self.lin2 = torch.nn.Linear(100,1)
        #self.lin3 = torch.nn.Linear(200,1)
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
arr1a = []
arr1b = []
arr2a = []
arr2b = []
for epoch in range(500000): 
  # TODO: fill in the blanks with your own code! 
    # resets the information from last time
    optimizer.zero_grad()
    # calculates the predictions
    pred_y = dl_model(ppg)
    pred_y_valid = dl_model(ppg_val)
    #print(pred_y)
    pred_y = pred_y.flatten()
    # calculates the loss
    loss = criterion(pred_y, bp)
    loss_valid = criterion(pred_y_valid, expected)
    # gradient descent, part 1
    loss.backward()
    # gradient descent, part 2
    optimizer.step()
    #plot
    if(epoch>1000):
      arr1a.append(epoch)
      arr1b.append(loss.item())
      arr2a.append(epoch)
      arr2b.append(loss_valid.item())

    # print loss every 500 epochs
    if epoch % 5000 == 0:
        #print(pred_y)
        print(f"{epoch}:{loss.item()}")
        print(f"{epoch}:{loss_valid.item()}")

'''
fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax1 = fig.add_subplot(2,1,1)

ax1.set_ylabel('Loss')
ax1.set_title('Training Set')
ax1.plot(arr1a,arr1b)

ax2 = fig.add_subplot(2,1,2)
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.set_title('Validation Set')
ax2.plot(arr2a,arr2b)

fig.savefig('epoch vs loss')

plt.show()
plt.plot(arr1a,arr1b)
plt.savefig('training loss')
plt.plot(arr1a,arr2b)
plt.savefig('validation loss')
torch.save(dl_model.state_dict(), 'state_dict')
'''



