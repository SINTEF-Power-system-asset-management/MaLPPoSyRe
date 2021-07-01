#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import torch
from torch_geometric.data import Data, Dataset, DataLoader
import pandas as pd
import numpy as np
import random as random
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
"""
set_up: Csv file containing the line capacity between the areas for each, used as edge weights in the graph
cost: The total cost values for each scenario defined as the deseried variable y.
"""
set_up = pd.read_csv('prev_line_caps.csv')
cost = pd.read_csv('tot_exp_cost.csv')


y_list =  []
for i in range(len(cost.iloc[0:,1])-1):
    y_list.append(cost['Total expected cost'].iloc[i])

split_value = 360
y_train = y_list[:split_value]
y_test = y_list[split_value:]

y_train = np.array(y_train)
y_train = y_train.reshape((len(y_train),1))
pt = PowerTransformer(method='box-cox', standardize=True)
y_train_transformed = pt.fit_transform(y_train)
y_train_transformed = y_train_transformed.reshape(-1)

y_test = np.array(y_test)
y_test = y_test.reshape((len(y_test),1))
y_test_transformed = pt.transform(y_test)
y_test_transformed = y_test_transformed.reshape(-1)


"""
We start by creating a dataset. This is done piecewise by creating a datapoint per scenario from set_up and cost.
data_list is the list containing all the datapoints. Each datapoint is a single graph. The goal is to construct
a list of datapoints and then create the datasets from those.
"""
training_set = []
test_set = []
#data_list = []
for i in range(len(cost.iloc[0:,1])-1):
    """
    Loop used to create the dataset
    
    Edge_attribute: Edge Feature matrix
    edge_index    : Graph connectivity in coo format, explains how each node is connected
    y             : The total expected cost value. The goal is to minimize this value
    data_list     : The list containing the datapoints
    
    This loop creates all the graphs and describes the topology and the features of the edges
    It also defines the features such as capacity of the edges.
    """
    if i < split_value:
        y = torch.Tensor([[y_train_transformed[i]]])
        x = torch.Tensor([[1],[1],[1]])
        edge_index = torch.Tensor([[0,0,1,1],[1,2,2,2]])
        edge_index = edge_index.type(torch.long)
        edge_attribute = torch.Tensor([set_up['Line10'].iloc[i + 1], set_up['Line11'].iloc[i + 1], set_up['Line18'].iloc[i + 1], set_up['Line19'].iloc[i + 1]])
        training_set.append(Data(x=x,edge_index=edge_index, edge_attr=edge_attribute, y=y))
    
    else:
        y = torch.Tensor([[y_test_transformed[i-split_value]]])
        x = torch.Tensor([[1],[1],[1]])
        edge_index = torch.Tensor([[0,0,1,1],[1,2,2,2]])
        edge_index = edge_index.type(torch.long)
        edge_attribute = torch.Tensor([set_up['Line10'].iloc[i + 1], set_up['Line11'].iloc[i + 1], set_up['Line18'].iloc[i + 1], set_up['Line19'].iloc[i + 1]])
        test_set.append(Data(x=x,edge_index=edge_index, edge_attr=edge_attribute, y=y))
    
    
    """
    #y = torch.Tensor([[cost['Total expected cost'].iloc[i]]])
    y = torch.Tensor([[y_normalized[i]]])
    x = torch.Tensor([[1],[1],[1]])
    edge_index = torch.Tensor([[0,0,1,1],[1,2,2,2]])
    edge_index = edge_index.type(torch.long)
    edge_attribute = torch.Tensor([set_up['Line10'].iloc[i + 1], set_up['Line11'].iloc[i + 1], set_up['Line18'].iloc[i + 1], set_up['Line19'].iloc[i + 1]])
    
    data_list.append(Data(x=x,edge_index=edge_index, edge_attr=edge_attribute, y=y))
    """
    

"""The data_list is then shuffled as to make the order of the datapoints random
"""
#random.shuffle(data_list)

"""The data_list is then split into two lists, a training set and a test set.
The splitting weight is done by a 80/20 ratio of training/test
The 360 value is found by 450 * 0.8 ~ 360
"""

training_dataset = DataLoader(training_set, batch_size=1)
test_dataset = DataLoader(test_set, batch_size=1)


"""
training_set = data_list[:360]
test_set = data_list[360:]
training_dataset = DataLoader(training_set, batch_size=1)
test_dataset = DataLoader(test_set, batch_size=1)
"""



# %%


torch.manual_seed(12345)
print(f'Number of training graphs: {len(training_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
#print(f'Number of test graphs: {training_dataset.num_node_features}')


# %%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear

"""
Here we create the Graph Convolutional Network Machine Learning Model.

The model is initalized by inheriting from the GCN module from Pytorch.
"""
class GCN(torch.nn.Module):
    def __init__(self, data_example, hidden_channels=8, dropout_rate=0):
        super(GCN, self).__init__()
        """
        Function used to initialize the convolution and linear layers.
        
        Args:
        data_example (graph)  : An example graph to set dimensions
        hidden_channels (int) : Amount of neurals in the hidden layers
        
        Creates:
        
        conv1 : First layer of the neural network.
        conv2 : Second layer of the neural network
        conv3 : Third layer of the neural network
        lin   : The final linear layer of the neural network.
        
        The Neural network has input size of the desired amount of node features.
        This is changed up / down to the amount of desired neurals in the hidden channels.
        Finally, in the linear layer the output is a single neuron which returns the desired 
        target value y.
        """
        self.conv1 = GCNConv(data_example.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.dropout_rate = dropout_rate
        
    def forward(self, data):
        """
        Forward function which passes the data through the neural network.
        
        Args:
            x           : Nodes
            edge_index  : Topology of the graph
            edge_weight : Weight of each edge
            
        For each pass through a layer, a Rectified Linear Unit (relu) is applied.
        A global_mean_pool function is applied to find the global mean in the 
        batch pool.
        Lastly, a dropout function is used to drop certain values to prevent
        overfitting
            
        """
        # 1. Obtain node embeddings
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr 
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index, edge_weight)
        #x = x.relu()
        #x = self.conv3(x, edge_index, edge_weight)
        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin(x)
        return x
    


# %%

data_example = next(iter(training_dataset))  # getting an example graph to get dimensions
model = GCN(data_example, hidden_channels=40, dropout_rate = 0.400)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    for data in training_dataset:  # Iterate in batches over the training dataset.
        out = model(data)
        #.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        loss = torch.sqrt(criterion(out, data.y))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    
def test(loader):
    
    model.eval()
    SE = 0
    for data in loader: # Iterate in batches over the training/test dataset
        out = model(data) #define the output value
        #print(data.batch)
        SE += (out - data.y)**2 #Sum over the SE
    RMSE = torch.sqrt(SE / len(loader.dataset))
    return RMSE # Finally we return the RMSE
     
def plotter(loader):
    
    model.eval()
    y_pred = []
    y_target = []
    x_values = []
    i = 0
    for data in loader:
        i += 1
        out = model(data)
        y_target.append(data.y)
        y_pred.append(out[0][0].detach().numpy())
        x_values.append(i)
    return y_pred, y_target, x_values


epoch_list = []
train_list = []
test_list = []
for epoch in range(1, 30):
    train()
    train_acc = test(training_dataset)
    test_acc = test(test_dataset)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc[0][0]:.4f}, Test Acc: {test_acc[0][0]:.4f}')
    epoch_list.append(epoch)
    train_list.append(train_acc[0][0].detach().numpy())
    test_list.append(test_acc[0][0].detach().numpy())



# %%
import matplotlib.pyplot as plt
y_pred, y_target, x_values = plotter(test_dataset)
plt.plot(x_values, y_pred, color='red', label='Predicted value')
plt.plot(x_values, y_target, color='green', label='Target Value')
plt.xlabel('Epoch')
plt.ylabel('Normalized Cost Value')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()



# %%
import matplotlib.pyplot as plt

plt.plot(epoch_list,train_list, color='blue', label='Training set')
plt.plot(epoch_list, test_list, color='red', label='Test set')
plt.xlabel('Epoch')
plt.ylabel('Relative Mean Square Error (RMSE)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
#plt.savefig("/Users/fredrikha/Workplace/graphconvolutional/Results GCN/Iteration 1/graph_3_hidden_layers.png")
plt.show()


# %%

# %%
