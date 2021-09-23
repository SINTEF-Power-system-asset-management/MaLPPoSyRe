# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
import random as random
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
    MinMaxScaler,
)
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split

# +
"""
Estimating costs linked to failure scenarios, preventive and corrective strategies.
data.

Set opf = True, to import file with power flow in lines instead of capacity
"""

import os

os.chdir("C:\my-proj\MaLPPoSyRe\data")


opf = True

if opf:
    df = pd.read_csv("tot_cost_failure_scenarios_140921.csv")
else:
    df = pd.read_csv("tot_cost_failure_scenarios_capacity_090921.csv")


"""
If set_max_cost = True, values higher than max_cost will be set to max_cost.
"""

set_max_cost = True

if set_max_cost:
    max_cost = df["Total cost"].min() * 3
    df.loc[df["Total cost"] > max_cost, "Total cost"] = max_cost


"""
train_test_split splits data into train and test while also shufling. test_size=0.2 puts 20% of data i test dataframe
random_state choses seed for shuffling which gives more consistency for our process.
"""

train, test = train_test_split(df, test_size=0.2, random_state=1)

edge_weights_train = train[["Line10", "Line11", "Line18", "Line19"]].values
edge_weights_test = test[["Line10", "Line11", "Line18", "Line19"]].values

y_train = train["Total cost"].values
y_test = test["Total cost"].values

""" 
Here, the user can choose which transformer to use by changing the name of variable transform_method. 
The transformation methods are:

PowerTransformer
QuantileTransformer
StandardScaler
MinMaxScaling (Normalization)

The training set is both fitted and transformed before it is reshaped back to its original form

It is also possible to run the script without any transformation. This is done by changing the 
transform_data parameter to False.
"""

transform_data = True
transform_method = "StandardScaler"


def transform_data_method(t_type):
    if t_type == "StandardScaler":
        return StandardScaler()
    elif t_type == "PowerTransformer":
        return PowerTransformer(method="box-cox", standardize=True)
    elif t_type == "QuantileTransformer":
        return QuantileTransformer(n_quantiles=5, random_state=0)
    elif t_type == "MinMaxScaler":
        return MinMaxScaler()


"""
Scaling cost, inter area flow and edge weights individually as a quick fix.
"""

if transform_data:
    pt = transform_data_method(transform_method)
    y_train = y_train.reshape((len(y_train), 1))
    y_train = pt.fit_transform(y_train)
    y_train = y_train.reshape(-1)

    y_test = y_test.reshape((len(y_test), 1))
    y_test = pt.transform(y_test)
    y_test = y_test.reshape(-1)

    sc = transform_data_method(transform_method)
    train["inter_area_flow"] = sc.fit_transform(
        train["inter_area_flow"].values.reshape(train.shape[0], 1)
    ).reshape(-1)
    test["inter_area_flow"] = sc.transform(
        test["inter_area_flow"].values.reshape(test.shape[0], 1)
    ).reshape(-1)

    lt = transform_data_method(transform_method)
    edge_weights_train = edge_weights_train.reshape((len(edge_weights_train) * 4, 1))
    edge_weights_train = lt.fit_transform(edge_weights_train)
    edge_weights_train = edge_weights_train.reshape(int(len(edge_weights_train) / 4), 4)

    edge_weights_test = edge_weights_test.reshape((len(edge_weights_test) * 4, 1))
    edge_weights_test = lt.transform(edge_weights_test)
    edge_weights_test = edge_weights_test.reshape(int(len(edge_weights_test) / 4), 4)


"""
Creating node attributes.
"""

node_attr_train = train[
    [
        "Area1-corr",
        "Area1-fault",
        "Area1-fault2",
        "inter_area_flow",
        "Area2-corr",
        "Area2-fault",
        "Area2-fault2",
        "inter_area_flow",
        "Area3-corr",
        "Area3-fault",
        "Area3-fault2",
        "inter_area_flow",
    ]
].values
node_attr_train = node_attr_train.reshape(int(len(node_attr_train)), 3, 4)

node_attr_test = test[
    [
        "Area1-corr",
        "Area1-fault",
        "Area1-fault2",
        "inter_area_flow",
        "Area2-corr",
        "Area2-fault",
        "Area2-fault2",
        "inter_area_flow",
        "Area3-corr",
        "Area3-fault",
        "Area3-fault2",
        "inter_area_flow",
    ]
].values
node_attr_test = node_attr_test.reshape(int(len(node_attr_test)), 3, 4)


# +
"""We start by creating a dataset. This is done piecewise by creating 
a datapoint per scenario. Each datapoint is a single graph 
The goal is to construct a list of datapoints and then create the datasets from those.
"""


def create_dataset(y_value, edge_weights, node_attr):
    """
    Loop used to create the dataset
    y_value       : The total expected cost value.
    edge_weights  : The edge weights based on OPF flow or capacity
    edge_attribute: Edge Feature matrix
    edge_index    : Graph connectivity in coo format, explains how each node is connected
    x             : Node feature, baseline set to 1 for each node.
    data_list     : The list containing the datapoints

    returns       : Graph and describes the topology and the features of the edges
                    It also defines the features such as capacity of the edges."""
    y = torch.Tensor([[y_value]])
    x = torch.Tensor(node_attr)
    edge_index = torch.Tensor([[0, 0, 1, 1], [1, 2, 2, 2]])
    edge_index = edge_index.type(torch.long)
    edge_attribute = torch.Tensor(edge_weights)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attribute, y=y)


"""
Map functions loop through all datapoints.
"""

training_set = list(map(create_dataset, y_train, edge_weights_train, node_attr_train))
test_set = list(map(create_dataset, y_test, edge_weights_test, node_attr_test))


training_dataset = DataLoader(training_set, batch_size=1)
test_dataset = DataLoader(test_set, batch_size=1)


print(f"Number of training graphs: {len(training_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")


"""Here we create the Graph Convolutional Network Machine Learning Model.

The model is initalized by inheriting from the GCN module from Pytorch.
"""


class GCN(torch.nn.Module):
    def __init__(self, data_example, hidden_channels=8, dropout_rate=0):
        super(GCN, self).__init__()
        """Function used to initialize the convolution and linear layers.
        
        Args:
        data_example (graph)  : An example graph to set dimensions
        hidden_channels (int) : Amount of neurals in the hidden layers
        dropout_rate (int)    : Rate of dropout
        
        Creates:
        
        conv1 : First layer of the neural network.
        conv2 : Optional Second layer of the neural network
        conv3 : Optional Third layer of the neural network
        lin   : The final linear layer of the neural network.
        
        The Neural network has input size of the desired amount of node features.
        This is changed up / down to the amount of desired neurals in the hidden channels.
        Finally, the output in the linear layer is a single neuron which returns the desired 
        target value y.
        """
        self.conv1 = GCNConv(
            data_example.num_node_features, hidden_channels, normalize=False
        )
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=False)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        """Forward function which passes the data through the neural network.

        Args:
            x           : Node-features
            edge_index  : Topology of the graph
            edge_weight : Weight of each edge

        For each passthrough of a layer, a Rectified Linear Unit (relu) is applied.
        A global_mean_pool function is applied to find the global mean in the
        batch pool.
        Lastly, a dropout function is used to drop certain values to prevent
        overfitting
        """
        # 1. Obtain node embeddings
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        y = self.conv1(x, edge_index, edge_weight)
        y = y.relu()
        # y = F.dropout(y, training=self.training)
        y = self.conv2(y, edge_index, edge_weight)
        y = y.relu()

        # y = self.conv3(y, edge_index, edge_weight)
        # y = y.relu()

        # 2. Readout layer
        y = global_mean_pool(y, data.batch)  # [batch_size, hidden_channels]
        y = F.dropout(y, p=self.dropout_rate, training=self.training)
        y = self.lin(y)
        return y


# +
data_example = next(
    iter(training_dataset)
)  # getting an example graph to get dimensions

"""Here the model, optimizer function, loss-criterion and the
learning-rate scheduler is initalized. 

model : Initalizes the neuron network
optimizer : Initializes the backpropagation optimizer
criterion : Defines the error function
scheduler : Initalizes the learning-rate reducter function

Disclaimer: Even though the MSELoss() criterion is called, in the train
function the square-root is taken off the loss-function, making
the MSE to the RMSE
"""
model = GCN(data_example, hidden_channels=12, dropout_rate=0.2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0018)

criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)  # Factor for learning rate decay


def train():
    """Training function which trains the model.
    The function iterates over the training set, estimates the y-value and
    compares it with the actual value. The RMSE and the gradients are then
    calculated and are fed back to the optimization function which
    updates the weights of the neural network.
    """
    model.train()
    for data in training_dataset:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Output value of the model
        loss = torch.sqrt(criterion(out, data.y))  # Compute the loss.
        # loss = criterion(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    scheduler.step()  # Reduction of learning rate


def test(loader):
    """Test function which evaluates the model by
    calculating the Root Mean Square Error

    Args:
    Loader - Dataset input, either training or test

    Creates:

    Root Mean Square Error between the:
    out - output value from the model
    data.y - actual value

    if Choice = 0, the function assume a transformation method
    was used and will try to inverse-transform the values before
    calculating the RMSE.
    If choice =/= 0, the RMSE is calculated directly as there
    hasn't been used a transformation method on the data.
    """
    if transform_data:
        model.eval()
        SE = 0
        for data in loader:  # Iterate in batches over the training/test dataset
            out = model(data)  # define the output value
            out = out.reshape(-1, 1)  # Reshape the size of the tensor
            out = pt.inverse_transform(
                out.detach().numpy()
            )  # Inverse transform the value
            y = data.y
            y = y.reshape(-1, 1)  # Reshaping the size of the tensor
            y = pt.inverse_transform(
                y.detach().numpy()
            )  # Inverse transforming the value
            SE += (out - y) ** 2  # Sum over the SE
        RMSE = np.sqrt(SE / len(loader.dataset))
        return RMSE  # Finally we return the RMSE
    else:
        model.eval()
        SE = 0
        for data in loader:  # Iterate in batches over the training/test dataset
            out = model(data)  # define the output value
            SE += (out - data.y) ** 2  # Sum over the SE
        RMSE = torch.sqrt(SE / len(loader.dataset))
        return RMSE  # Finally we return the RMSE


def plotter(loader):
    """Plotter Function which returns the predicted value,
    the target value and the x value which can be used for plotting
    after the model has been trained.
    Args:

    Loader - Dataset input, either training or test

    Creates:

    y_target - target value
    y_pred - predicted value from the model
    x_values - Scenario value. (p1, p11, p111 etc)
    """
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


# -

"""Below is a loop which trains the model over a set amount of epochs and 
which prints the current epoch and the RMSE values for the training-set and the test-set.
The RMSE values and the epoch values are stored in seperate lists.
In addition, a breakpoint has been defined so that if the RMSE is less than a 
desired value between two epochs, the training stops to prevent overfitting.
"""
if transform_data:
    epoch_list = []
    train_list = []
    test_list = []
    previous_rmse_test = 0
    for epoch in range(1, 100):
        train()
        train_acc = test(training_dataset)
        test_acc = test(test_dataset)
        current_rmse_test = test_acc[0][0]
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc[0][0]:.4f}, Test Acc: {test_acc[0][0]:.4f}"
        )
        epoch_list.append(epoch)
        train_list.append(train_acc[0][0])
        test_list.append(test_acc[0][0])
        if abs(previous_rmse_test - current_rmse_test) < 0.00001:
            print(f"solution converged at epoch: {epoch}")
            break
        else:
            previous_rmse_test = current_rmse_test
else:
    epoch_list = []
    train_list = []
    test_list = []
    previous_rmse_test = 0
    for epoch in range(1, 100):
        train()
        train_acc = test(training_dataset)
        test_acc = test(test_dataset)
        current_rmse_test = test_acc[0][0]
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc[0][0]:.4f}, Test Acc: {test_acc[0][0]:.4f}"
        )
        epoch_list.append(epoch)
        train_list.append(train_acc[0][0].detach().numpy())
        test_list.append(test_acc[0][0].detach().numpy())
        if abs(previous_rmse_test - current_rmse_test) < 0.000001:
            print(f"solution converged at epoch: {epoch}")
            break
        else:
            previous_rmse_test = current_rmse_test

"""Plot showing the difference between the target value and the predicted value
for the training set
"""
y_pred, y_target, x_values = plotter(training_dataset)
plt.plot(x_values, y_target, color="green", label="Target Value")
plt.plot(x_values, y_pred, color="red", label="Predicted value")
plt.xlabel("Scenario")
plt.ylabel("Cost Value")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncol=2,
    mode="expand",
    borderaxespad=0.0,
)
plt.show()

"""Plot showing the difference between the target value and the predicted value
for the test set
"""
y_pred, y_target, x_values = plotter(test_dataset)
plt.plot(x_values, y_target, color="green", label="Target Value")
plt.scatter(x_values, y_pred, color="red", label="Predicted value", s=25)
plt.xlabel("Scenario")
plt.ylabel("Cost Value")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncol=2,
    mode="expand",
    borderaxespad=0.0,
)
plt.show()


"""Plot showing the difference between the target value and the predicted value
for the test set, target data in ascending order
"""
y_pred, y_target, x_values = plotter(test_dataset)

plot_df = pd.DataFrame(data={"pred": y_pred, "target": y_target})
plot_df.sort_values(by="target", inplace=True)


plt.plot(x_values, plot_df["pred"], color="red", label="Predicted value")
plt.plot(x_values, plot_df["target"], color="green", label="Target Value")
plt.xlabel("Scenario")
plt.ylabel("Cost Value")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncol=2,
    mode="expand",
    borderaxespad=0.0,
)
plt.show()


"""Plot showing the change of RMSE for both the training set and 
the test set.
"""

plt.plot(epoch_list, train_list, color="blue", label="Training set")
plt.plot(epoch_list, test_list, color="red", label="Test set")
plt.xlabel("Epoch")
plt.ylabel("Root Mean Square Error (RMSE)")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncol=2,
    mode="expand",
    borderaxespad=0.0,
)
plt.show()


# +
"""Following is the hyperparameter-optimization loop done following the random search method.

The loop creates 200 models, each with a random learning rate, hidden channels number
and dropout rate number taken from a predefined range. 
The loop then starts to train the model over 100 epochs, or 
until the model has converged and the change in RMSE is lower than a set threshold.

The RMSE value and all the other hyperparameters are stored in each their separate list,
and when the loop has been iterated over, a second loop goes over the RMSE list and finds the
hyperparameter-settings with the lowest RMSE values.
"""
min_rmse_train_list = []
min_rmse_test_list = []
lr_list = []
hidden_channels_list = []
dropout_rate_list = []
epoch_list = []

for i in range(50):
    lr = random.randrange(1, 101) / 10000  # Here we find a random learnig rate value
    hidden_channels = random.randrange(
        5, 31
    )  # Here we find a random hidden_channel value in the range of 5-50
    dropout_rate = (
        random.randrange(0, 10) / 20
    )  # Here we find a random dropout_rate value in the range of 0 and 1
    # epoch = random.randrange(10,50) #Here we find a random epoch value in the range of 10 and 50
    epoch = 150

    model = GCN(
        data_example, hidden_channels=hidden_channels, dropout_rate=dropout_rate
    )  # Here we initalize our GCN model
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # We define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = (
        torch.nn.MSELoss()
    )  # And we define the criterion-type, Mean square error loss in our case.

    min_rmse_train = 0
    min_rmse_test = 0
    previous_rmse_test = 0
    for i in range(epoch):
        train()
        train_acc = test(training_dataset)
        test_acc = test(test_dataset)
        current_rmse_train = train_acc[0][0]
        current_rmse_test = test_acc[0][0]
        """Here the min values are initialized if this is the first time through the loop.
        """
        if min_rmse_train == 0 and min_rmse_test == 0:
            min_rmse_train = current_rmse_train
            min_rmse_test = current_rmse_test
        """Here the minimum value for the train & test rmse is changed if the current value is smaller than the current min value
        """
        if current_rmse_train < min_rmse_train:
            min_rmse_train = current_rmse_train

        if current_rmse_test < min_rmse_test:
            min_rmse_test = current_rmse_test
        """Here I create the option to break out of the loop if the solution converges
        """
        if abs(previous_rmse_test - current_rmse_test) < 0.00001:
            print(f"solution converged at epoch: {i}")
            break
        else:
            previous_rmse_test = current_rmse_test

            if i == 149:
                print(f"solution finally converged at epoch: {i}")
            continue
        break

    min_rmse_train_list.append(min_rmse_train)
    min_rmse_test_list.append(min_rmse_test)
    lr_list.append(lr)
    hidden_channels_list.append(hidden_channels)
    dropout_rate_list.append(dropout_rate)
    epoch_list.append(epoch)

# +

for i in range(len(lr_list)):
    print(
        f"Min MSE Test: {min_rmse_test_list[i]:.7f}, Min MSE Train: {min_rmse_train_list[i]:.7f}"
    )
    print(
        f"obtained with a learning rate of: {lr_list[i]:.4f}, a hidden channel nr of: {hidden_channels_list[i]:03d}"
    )
    print(
        f"a dropout rate of: {dropout_rate_list[i]:.4f} and an epoch number of: {epoch_list[i]:03d}"
    )


"""
Here I find the index of the minimum value of the RMSE. This is done this way
as pytorch don't have a method of doing it.
"""

value = 0
min_rmse = 0
for i in range(len(min_rmse_test_list)):
    if min_rmse == 0:
        min_rmse = min_rmse_test_list[i]

    if min_rmse_test_list[i] < min_rmse:
        min_rmse = min_rmse_test_list[i]
        value = i

print(min_rmse)
print(value)

print(
    f"The lowest total MSE value, {min_rmse_test_list[value]:.7f}, for the test set was found with the following settings: "
)
print(
    f"a learning rate of: {lr_list[value]:.4f}, a hidden channel nr of: {hidden_channels_list[value]:03d}"
)
print(
    f"a dropout rate of: {dropout_rate_list[value]:.4f} and an epoch number of: {epoch_list[value]:03d}"
)
# -
