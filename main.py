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
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD


from model import DeepNeuralNetwork



model = DeepNeuralNetwork()
model.load_state_dict(torch.load('state_dict'))
model.eval()
valid = pd.read_csv('ppg_valid.csv')
ppg_val = torch.tensor(valid[['max1', 'min1', 'max2','min2', 'max3', 'min3']].values,dtype=torch.float32)

expected = torch.tensor(valid['dbp'].values,dtype = torch.float32)
predicted = model(ppg_val)
np_arr_exp = expected.cpu().detach().numpy()
np_arr_pred = predicted.cpu().detach().numpy()
f = open("expected.txt", "w")
f.write(str(np_arr_exp))
f.close()
g = open("predicted.txt", "w")
g.write(str(np_arr_pred))
g.close()
errors = mean_squared_error(np_arr_exp, np_arr_pred)
print(errors)
#print(model(torch.tensor([2368.000,1680.000,2432.000,1677.000,2235.000,1861.000])))
#print(model(torch.tensor([3931.000,1670.000,3775.000,1638.000,3943.000,1859.000])))