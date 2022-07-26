import div_methods as div
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn import Linear, Sequential, Sigmoid
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, NNConv
import random as random


df = pd.read_csv("data/tot_cost_failure_scenarios_041021_multicorr2.csv")


"""
If set_max_cost = True, values higher than max_cost will be removed
Or values higher than set_max can be set to set_max
"""

set_max_cost = True

if set_max_cost:
    max_cost = 2
    # df = df[df["Total cost"] < max_cost]
    df.loc[df["Total cost"] > max_cost, "Total cost"] = max_cost


df["fault_Line10"] = df["fault"].apply(lambda row: 1 if row == "fault_L10" else 0)
df["fault_Line11"] = df["fault"].apply(lambda row: 1 if row == "fault_L11" else 0)
df["fault_Line18"] = df["fault"].apply(lambda row: 1 if row == "fault_L18" else 0)
df["fault_Line19"] = df["fault"].apply(lambda row: 1 if row == "fault_L19" else 0)


"""
train_test_split splits data into train and test while also shuffling. test_size=0.2 puts 
20% of data in test dataframe random_state chooses seed for shuffling which gives more 
consistency for our process.
"""

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

edge_features_train = abs(
    df_train[
        [
            "Line10",
            "fault_Line10",
            "Line11",
            "fault_Line11",
            "Line18",
            "fault_Line18",
            "Line19",
            "fault_Line19",
        ]
    ].values
)
edge_features_test = abs(
    df_test[
        [
            "Line10",
            "fault_Line10",
            "Line11",
            "fault_Line11",
            "Line18",
            "fault_Line18",
            "Line19",
            "fault_Line19",
        ]
    ].values
)

y_train = df_train["Total cost"].values
y_test = df_test["Total cost"].values

""" 
Here, the user can choose which transformer to use by changing the name of variable 
transform_method. The transformation methods are:

PowerTransformer
QuantileTransformer
StandardScaler
MinMaxScaler (Normalization)
"""

transform_method = "MinMaxScaler"

"""
Scaling cost, node and edge features separately as a quick fix.
"""

scale_y = div.transform_data_method(transform_method)
y_train = y_train.reshape((len(y_train), 1))
y_train = scale_y.fit_transform(y_train)
y_train = y_train.reshape(-1)

y_test = y_test.reshape((len(y_test), 1))
y_test = scale_y.transform(y_test)
y_test = y_test.reshape(-1)

scale_node = div.transform_data_method(transform_method)
train_node_scaled = scale_node.fit_transform(
    df_train[["inter_area_flow", "Area1_prod", "Area2_prod", "Area3_prod"]]
)
test_node_scaled = scale_node.transform(
    df_test[["inter_area_flow", "Area1_prod", "Area2_prod", "Area3_prod"]]
)

scale_edge = div.transform_data_method(transform_method)
edge_features_train = edge_features_train.reshape((len(edge_features_train) * 8, 1))
edge_features_train = scale_edge.fit_transform(edge_features_train)
edge_features_train = edge_features_train.reshape(
    int(len(edge_features_train) / 8), 4, 2
)

edge_features_test = edge_features_test.reshape((len(edge_features_test) * 8, 1))
edge_features_test = scale_edge.transform(edge_features_test)
edge_features_test = edge_features_test.reshape(int(len(edge_features_test) / 8), 4, 2)


"""
Creating node attributes.
"""

df_train["inter_area_flow_scaled"] = train_node_scaled[:, 0]
df_train["Area1_prod_scaled"] = train_node_scaled[:, 1]
df_train["Area2_prod_scaled"] = train_node_scaled[:, 2]
df_train["Area3_prod_scaled"] = train_node_scaled[:, 3]

df_test["inter_area_flow_scaled"] = test_node_scaled[:, 0]
df_test["Area1_prod_scaled"] = test_node_scaled[:, 1]
df_test["Area2_prod_scaled"] = test_node_scaled[:, 2]
df_test["Area3_prod_scaled"] = test_node_scaled[:, 3]

node_attr_train = df_train[
    [
        "Area1-corr",
        "Area1_prod_scaled",
        "inter_area_flow_scaled",
        "Area2-corr",
        "Area2_prod_scaled",
        "inter_area_flow_scaled",
        "Area3-corr",
        "Area3_prod_scaled",
        "inter_area_flow_scaled",
    ]
].values
node_attr_train = node_attr_train.reshape(int(len(node_attr_train)), 3, 3)

node_attr_test = df_test[
    [
        "Area1-corr",
        "Area1_prod_scaled",
        "inter_area_flow_scaled",
        "Area2-corr",
        "Area2_prod_scaled",
        "inter_area_flow_scaled",
        "Area3-corr",
        "Area3_prod_scaled",
        "inter_area_flow_scaled",
    ]
].values
node_attr_test = node_attr_test.reshape(int(len(node_attr_test)), 3, 3)


"""
We start by creating a dataset. This is done piecewise by creating 
a datapoint per scenario. Each datapoint is a single graph.
The goal is to construct a list of datapoints and then create the datasets from those.

Map functions loop through all datapoints.
"""

training_set = list(
    map(div.create_dataset, y_train, edge_features_train, node_attr_train)
)
test_set = list(map(div.create_dataset, y_test, edge_features_test, node_attr_test))


"""
Can change batch size. At the moment batch_size != 1 not working.

"""

batch_size = 1
training_dataset = DataLoader(training_set, batch_size=batch_size)
test_dataset = DataLoader(test_set, batch_size=batch_size)

print(f"Number of training graphs: {len(training_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")


"""
Here we create the Graph Convolutional Network Machine Learning Model.

The model is initalized by inheriting from the NNConv from Pytorch and uses Sequential as 
the edge function. Will later move this class to div_methods.py
"""


class GCN(torch.nn.Module):
    def __init__(
        self, data_example, hidden_channels=8, dropout_rate=0, isthird_layer=False
    ):
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

        self.conv1 = NNConv(
            data_example.num_node_features, hidden_channels, self.edge_func, "mean"
        )
        self.conv2 = NNConv(hidden_channels, hidden_channels, self.edge_func, "mean")
        self.conv3 = NNConv(hidden_channels, hidden_channels, self.edge_func, "mean")
        self.lin = Linear(hidden_channels, 1)
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.isthird_layer = isthird_layer

    def edge_func(self, edge_feat):
        # lin = Linear(data_example.num_edge_features, data_example.num_node_features*self.hidden_channels)
        lin = Sequential(
            Linear(data_example.num_edge_features, self.hidden_channels),
            Sigmoid(),
            Linear(
                self.hidden_channels,
                data_example.num_node_features * self.hidden_channels,
            ),
        )
        return lin(edge_feat)

    def forward(self, data):
        """Forward function which passes the data through the neural network.

        Args:
            x           : Node-features
            edge_index  : Topology of the graph
            edge_features : features of each edge

        For each passthrough of a layer, a Rectified Linear Unit (relu) is applied.
        A global_mean_pool function is applied to find the global mean in the
        batch pool.
        Lastly, a dropout function is used to drop certain values to prevent
        overfitting
        """
        # 1. Obtain node embeddings
        x, edge_index, edge_features = data.x, data.edge_index, data.edge_attr

        y = self.conv1(x, edge_index, edge_features)
        y = y.relu()
        # y = F.dropout(y, training=self.training)
        y = self.conv2(y, edge_index, edge_features)
        y = y.relu()

        if self.isthird_layer:
            y = self.conv3(y, edge_index, edge_features)
            y = y.relu()

        # 2. Readout layer
        y = global_mean_pool(y, data.batch)  # [batch_size, hidden_channels]
        y = F.dropout(y, p=self.dropout_rate, training=self.training)
        y = self.lin(y)
        return y


# +
data_example = next(
    iter(training_dataset)
)  # getting an example graph to get dimensions


"""
Here the model, optimizer function, loss-criterion and the
learning-rate scheduler is initalized. 

model : Initalizes the neuron network
optimizer : Initializes the backpropagation optimizer
scheduler : Initalizes the learning-rate reducter function
criterion : Defines the error function
is_rmse : Boolean set to True if want RMSE as criterion. 
          Important that Criterion is set to MSELoss if this is used
"""

model = GCN(data_example, hidden_channels=12, dropout_rate=0.2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0035)


scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)  # Factor for learning rate decay

criterion = torch.nn.MSELoss()
is_rmse = True


"""
Below is a loop which trains the model over a set amount of epochs and 
which prints the current epoch and the RMSE values for the training-set and the test-set.
The train and test errors are stored in different arrays.
In addition, a breakpoint has been defined so that if the error is less than a 
desired value between two epochs, the training stops to prevent overfitting.

num_epochs : Number of epochs in training
"""


num_epochs = 100
train_list = np.array([])
test_list = np.array([])
previous_loss_test = 0
for epoch in range(1, num_epochs + 1):
    div.train_model(
        model, training_dataset, optimizer, scheduler, criterion, is_rmse=is_rmse
    )
    train_acc = div.test_model(
        model, training_dataset, batch_size=batch_size, scale=scale_y, calc_rmse=is_rmse
    )
    test_acc = div.test_model(
        model, test_dataset, batch_size=batch_size, scale=scale_y, calc_rmse=is_rmse
    )
    current_loss_test = test_acc
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.7f}, Test Acc: {test_acc:.7f}")
    train_list = np.concatenate((train_list, train_acc), axis=None)
    test_list = np.concatenate((test_list, test_acc), axis=None)
    if abs(previous_loss_test - current_loss_test) < 0.00001:
        print(f"solution converged at epoch: {epoch}")
        break
    else:
        previous_loss_test = current_loss_test


"""
Plot showing the difference between the target value and the predicted value
for the test set, target data in ascending order.
"""

y_pred, y_target, x_values = div.plotter(model, test_dataset)

plot_df = pd.DataFrame(data={"pred": y_pred, "target": y_target})
plot_df.sort_values(by="target", inplace=True)

plt.scatter(x_values, plot_df["pred"], color="red", label="Predicted value", s=1)
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


"""
Plot showing the change of error for both the training set and the test set.
"""

epoch_list = np.arange(1, len(train_list) + 1)
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
