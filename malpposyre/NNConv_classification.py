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

# +
"""
Classifying costs linked to failure scenarios, preventive and corrective strategies.
data.

Import file with scenarios as a dataframe.

prev_id: id of preventive strategy
Line10, Line11, Line18, Line19: Power flow in respetive lines
Area1 prod, Area2 prod, Area3 prod: total production in respective areas
Area1 load, Area2 load, Area3 load: total load in respective areas
inter_area_flow: inter-area flow
fault: which line has fault in given scenario
corr: which area has corrective measure (load or production disconnected)
Total cost: Total cost of system operation for given scenario

Rest of columns are One hot encodings. Unsure if we need to differentiate between disconnected 
load and production since we only have 3 possible corrective measures. ??
"""

df = pd.read_csv("data/tot_cost_failure_scenarios_041021_multicorr2.csv")

# +
"""
Scenarios with total cost less than thresh set to 0. Scenarios with total cost
more than thresh set to 1.
"""

thresh = 1.126
df["Class"] = df.apply(lambda row: 0.0 if row["Total cost"] < thresh else 1.0, axis=1)


# Adding edge features
df["fault_Line10"] = df["fault"].apply(lambda row: 1 if row == "fault_L10" else 0)
df["fault_Line11"] = df["fault"].apply(lambda row: 1 if row == "fault_L11" else 0)
df["fault_Line18"] = df["fault"].apply(lambda row: 1 if row == "fault_L18" else 0)
df["fault_Line19"] = df["fault"].apply(lambda row: 1 if row == "fault_L19" else 0)

# +
"""
train_test_split splits data into train and test while also shuffling. test_size=0.2 puts 20% of data i test dataframe
random_state chooses seed for shuffling which gives more consistency for our process.

When access to more data, should split into train, validation and test-set. ??

Splitting into train and test set while keeping all scenarios with the same prev_id in the same set.
Maybe not necessary in classification model.
"""

prev_ids = df.prev_id.unique()

grouped = df.groupby(df.prev_id)

df_list = []
for prev_id in prev_ids:
    df_list.append(grouped.get_group(prev_id))


train_df_list, test_df_list = train_test_split(df_list, test_size=0.2, random_state=1)

df_train = pd.concat(train_df_list)
df_test = pd.concat(test_df_list)


# +
"""
Create ege features.
"""

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

y_train = df_train["Class"].values
y_test = df_test["Class"].values

# +
""" 
Here, the user can choose which transformer to use by changing the name of variable 
transform_method. The transformation methods are:

PowerTransformer
QuantileTransformer
StandardScaler
MinMaxScaler (Normalization)
"""

transform_method = "MinMaxScaler"

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

# +
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

# +
"""
We start by creating a dataset. This is done piecewise by creating 
a datapoint per scenario. Each datapoint is a single graph.
The goal is to construct a list of datapoints and then create the datasets from those.
"""

# Map function loops through all datapoints.
training_set = list(
    map(div.create_dataset, y_train, edge_features_train, node_attr_train)
)
test_set = list(map(div.create_dataset, y_test, edge_features_test, node_attr_test))

training_dataset = DataLoader(training_set, batch_size=1)
test_dataset = DataLoader(test_set, batch_size=1)

print(f"Number of training graphs: {len(training_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")


# +
"""
Here we create the Graph Convolutional Network Machine Learning Model.

The model is initalized by inheriting from the NNConv from Pytorch and Sequential as 
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
        isthird_layer : Boolean. If True, third convolutional layer added

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
            Linear(data_example.num_edge_features, 4),
            Sigmoid(),
            Linear(4, data_example.num_node_features * self.hidden_channels),
        )
        return lin(edge_feat)

    def forward(self, data):
        """Forward function which passes the data through the neural network.

        Args:
            x           : Node-features
            edge_index  : Topology of the graph
            edge_features : Feature of each edge

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
        y = torch.sigmoid(self.lin(y))

        return y


# +
"""
Here the model, optimizer function, loss-criterion and the
learning-rate scheduler is initalized. 

model : Initalizes the neuron network
optimizer : Initializes the backpropagation optimizer
scheduler : Initalizes the learning-rate reducter function
criterion : Defines the error function
is_third_layer: Boolean. If set to True, model will use 3 GCN layers. 
                Else, model will use 2 GCN-layers.
"""

is_third_layer = False

# Hyperparameters: Set based on hyperparameteroptimisation.
hidden_channels = 12
dropout_rate = 0.2
learning_rate = 0.0005

# Getting an example graph to get dimensions
data_example = next(iter(training_dataset))

model = GCN(
    data_example,
    hidden_channels=hidden_channels,
    dropout_rate=dropout_rate,
    isthird_layer=is_third_layer,
)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)  # Factor for learning rate decay

# Weight based on the ratio between the majority and the minority class
pos_weight = 3
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

# +
"""
Below is a loop which trains the model over a set amount of epochs and 
which prints the current epoch and the accuracy for the training-set and the test-set.
The accuracy and the epoch values are stored in separate arrays.

num_epochs : Number of epochs in training
"""
num_epochs = 50
patience_value = 0.0001
patience_num = 5

train_list = np.array([])
test_list = np.array([])
acc_diff = np.array([])
previous_acc_test = 0
for epoch in range(1, num_epochs + 1):
    div.train_model(model, training_dataset, optimizer, scheduler, criterion)
    train_acc = div.test_model(
        model, training_dataset, batch_size=batch_size, classification=True
    )
    test_acc = div.test_model(
        model, test_dataset, batch_size=batch_size, classification=True
    )
    current_acc_test = test_acc
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    train_list = np.concatenate((train_list, train_acc), axis=None)
    test_list = np.concatenate((test_list, test_acc), axis=None)
    acc_diff = np.concatenate(
        (acc_diff, abs(previous_acc_test - current_acc_test)), axis=None
    )
    if acc_diff[-patience_num + 1 :].mean() < patience_value:
        print(f"solution converged at epoch: {epoch}")
        break
    else:
        previous_acc_test = current_acc_test


# +
"""
df_results return results for test data
"""

y_pred, y_prob, y_target, x_values = div.plotter(model, test_dataset)

df_results = df_test.copy()
df_results["target"] = y_target[0]
df_results["pred"] = y_pred[0]
df_results["prob"] = y_prob[0]
# -
