import torch
import torch.nn.functional as f
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
    MinMaxScaler,
)


class GCN(torch.nn.Module):
    def __init__(self, data_example, hidden_channels=8, dropout_rate=0, isthird_layer=False, is_classification=False):
        super(GCN, self).__init__()
        """
        Here we create the Graph Convolutional Network Machine Learning Model. The model is initalized by inheriting 
        from the GCN module from Pytorch. 
        
        Function used to initialize the convolution and linear layers.

        Args:
        data_example (graph)  : An example graph to set dimensions
        hidden_channels (int) : Amount of neurals in the hidden layers
        dropout_rate (int)    : Rate of dropout
        isthird_layer (boolean) : If true adds third layer to the neural network.
        is_classification (boolean) : If true uses Sigmoid function instead of Linear as last layer.
        
        Creates:

        conv1 : First layer of the neural network.
        conv2 : Second layer of the neural network.
        conv3 : Optional Third layer of the neural network.
        lin   : The final linear layer of the neural network.

        The Neural network has input size of the desired amount of node features.
        This is changed up / down to the amount of desired neurals in the hidden channels.
        Finally, the output in the linear layer is a single neuron which returns the desired 
        target value y.
        """
        self.conv1 = GCNConv(data_example.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.dropout_rate = dropout_rate
        self.third_layer = isthird_layer
        self.is_classification = is_classification

    def forward(self, data):
        """Forward function which passes the data through the neural network.

        Args:
            data.x           : Node-features
            data.edge_index  : Topology of the graph
            data.edge_weight : Weight of each edge

        For each passthrough of a layer, a Rectified Linear Unit (relu) is applied.
        A global_mean_pool function is applied to find the global mean in the
        batch pool.
        Lastly, a dropout function is used to drop certain values to prevent
        overfitting.
        """
        # 1. Obtain node embeddings
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        y = self.conv1(x, edge_index, edge_weight)
        y = y.relu()

        y = self.conv2(y, edge_index, edge_weight)
        y = y.relu()

        if self.third_layer:
            y = self.conv3(y, edge_index, edge_weight)
            y = y.relu()

        # 2. Readout layer
        y = global_mean_pool(y, data.batch)  # [batch_size, hidden_channels]
        y = f.dropout(y, p=self.dropout_rate, training=self.training)

        if self.is_classification:
            y = torch.sigmoid(self.lin(y))
        else:
            y = self.lin(y)
        return y


def transform_data_method(t_type):
    if t_type == "StandardScaler":
        return StandardScaler()
    elif t_type == "PowerTransformer":
        return PowerTransformer(method="box-cox", standardize=True)
    elif t_type == "QuantileTransformer":
        return QuantileTransformer(n_quantiles=5, random_state=0)
    elif t_type == "MinMaxScaler":
        return MinMaxScaler()


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


def plotter(model, loader, is_classification=False):
    """Plotter Function which returns the predicted value,
    the target value and the x value which can be used for plotting
    after the model has been trained.
    Args:

    model - GCN model
    loader - Dataset input, either training or test
    is_classification - boolean. True if binary classification problem.

    Creates:

    y_target - target value
    y_pred - predicted value from the model
    scenarios - Scenario value
    """
    model.eval()
    y_pred = np.array([])
    y_target = np.array([])
    for data in loader:
        out = model(data)
        if is_classification:
            out = torch.Tensor([1 if x > 0.5 else 0 for x in out])
        y_target = np.concatenate((y_target, data.y.reshape(-1)), axis=None)
        y_pred = np.concatenate((y_pred, out.detach().numpy()), axis=None)

    scenarios = np.arange(1, len(y_pred) + 1)
    return y_pred, y_target, scenarios


def train_model(model, training_dataset, optimizer, scheduler, criterion, calc_rmse=False):
    """Training function which trains the model.
    The function iterates over the training set, estimates the y-value and
    compares it with the actual value. The accuracy and the gradients are then
    calculated and are fed back to the optimization function which
    updates the weights of the neural network.
    """
    model.train()
    for data in training_dataset:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Output value of the model
        loss = criterion(out, data.y)
        if calc_rmse:
            loss = torch.sqrt(criterion(out, data.y))
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    scheduler.step()  # Reduction of learning rate


def test_model(model, loader, batch_size=1, scale=None, transform_data=True, calc_rmse=False, classification=False):
    """Test function which evaluates the model by
    calculating the MSE og RMSE

    Args:
    model - GCN model
    loader - Dataset input, either training or test
    batch_size - batch size fed in to the neural network
    scale - if transformation is used add scaler
    transform_data - boolean. Set to true if transformation is used
    calc_rmse - boolean, only used if regression. calculates RMSE if set to True, else MSE
    classification - boolean. Set to True if classification problem.

    Creates:

    Error between the:
    out - output value from the model
    data.y - actual value
    """
    if classification:
        model.eval()
        TP = 0
        for data in loader:  # Iterate in batches over the training/test dataset
            out = model(data)  # define the output value
            #out = out.reshape(-1, 1)  # Reshape the size of the tensor
            pred = torch.Tensor([1 if x > 0.5 else 0 for x in out])
            pred = pred.reshape(-1, 1)
            y = data.y
            y = y.reshape(-1, 1)  # Reshaping the size of the tensor
            TP += int((pred == y).sum())  # Check against ground-truth labels.
        acc = TP / len(loader.dataset)
        return acc  # Finally we return the loss

    if transform_data:
        model.eval()
        SE = 0
        for data in loader:  # Iterate in batches over the training/test dataset
            out = model(data)  # define the output value
            out = out.reshape(-1, 1)  # Reshape the size of the tensor
            out = scale.inverse_transform(
                out.detach().numpy()
            )  # Inverse transform the value
            y = data.y
            y = y.reshape(-1, 1)  # Reshaping the size of the tensor
            y = scale.inverse_transform(
                y.detach().numpy()
            )  # Inverse transforming the value
            SE += ((out - y) ** 2).sum()  # Sum over the SE
        MSE = SE / (len(loader.dataset))
    else:
        model.eval()
        SE = 0
        for data in loader:  # Iterate in batches over the training/test dataset
            out = model(data)  # define the output value
            SE += ((out - data.y) ** 2).sum()  # Sum over the SE
        MSE = SE / (len(loader.dataset))

    if calc_rmse:
        return np.sqrt(MSE)
    return MSE
