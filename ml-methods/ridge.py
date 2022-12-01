import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
import numpy as np  
import torch
import torch.nn as nn
import pandas as pd                    
import matplotlib.pyplot as plt        
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import csv

dataset = pd.read_csv('data.csv', names = ['a', 'b','c','d','e','f','g','h','i','j', 'k','l','m','n','o','p','q','r','s'])
params = dataset[['a','b','c','f','g','h','i','j', 'k','l','m','n','o','p','q','r','s']]
map = list()
ss=StandardScaler()

with open('data.csv', 'r') as csvfile:
	parse = csv.reader(csvfile, delimiter = ',')
	for row in parse:
		map.append((2*float(row[3])+float(row[4]))/3)

x_train, x_test, y_train, y_test = train_test_split(params, map, test_size=0.33, random_state=0) 
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

reg = Ridge(normalize=True)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print('MAE', MAE)  
print('MSE:', MSE)  
print('RMSE:', RMSE)