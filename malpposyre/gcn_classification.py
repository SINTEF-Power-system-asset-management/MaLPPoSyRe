import torch
from torch_geometric.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import div_methods as div
import numpy as np

"""
Estimating costs linked to failure scenarios, preventive and corrective strategies.
data.
"""

df = pd.read_csv(r'data\tot_cost_failure_scenarios_041021_multicorr2.csv')

thresh = 1.5
df["Total cost"][df["Total cost"] < thresh] = 0
df["Total cost"][df["Total cost"] >= thresh] = 1

"""
train_test_split splits data into train and test while also shufling. test_size=0.2 puts 20% of data i test dataframe
random_state choses seed for shuffling which gives more consistency for our process.
"""

train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

"""
Important to have absolute value of power flow.
"""

edge_weights_train = abs(train_df[["Line10", "Line11", "Line18", "Line19"]].values)
edge_weights_test = abs(test_df[["Line10", "Line11", "Line18", "Line19"]].values)

y_train = train_df["Total cost"].values
y_test = test_df["Total cost"].values

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

transform_method = "MinMaxScaler"

"""
Scaling cost, inter area flow and edge weights individually as a quick fix. 
"""

scale_y = div.transform_data_method(transform_method)
y_train = y_train.reshape((len(y_train), 1))
y_train = scale_y.fit_transform(y_train)
y_train = y_train.reshape(-1)

y_test = y_test.reshape((len(y_test), 1))
y_test = scale_y.transform(y_test)
y_test = y_test.reshape(-1)

scale_node = div.transform_data_method(transform_method)
train_node_scaled = scale_node.fit_transform(train_df[["inter_area_flow", "Area1_prod", "Area2_prod", "Area3_prod"]])
test_node_scaled = scale_node.transform(test_df[["inter_area_flow", "Area1_prod", "Area2_prod", "Area3_prod"]])

scale_edge = div.transform_data_method(transform_method)
edge_weights_train = edge_weights_train.reshape((len(edge_weights_train) * 4, 1))
edge_weights_train = scale_edge.fit_transform(edge_weights_train)
edge_weights_train = edge_weights_train.reshape(int(len(edge_weights_train) / 4), 4)

edge_weights_test = edge_weights_test.reshape((len(edge_weights_test) * 4, 1))
edge_weights_test = scale_edge.transform(edge_weights_test)
edge_weights_test = edge_weights_test.reshape(int(len(edge_weights_test) / 4), 4)


"""
Creating node attributes.
"""

train_df["inter_area_flow_scaled"] = train_node_scaled[: ,0]
train_df["Area1_prod_scaled"] = train_node_scaled[: ,1]
train_df["Area2_prod_scaled"] = train_node_scaled[: ,2]
train_df["Area3_prod_scaled"] = train_node_scaled[: ,3]

test_df["inter_area_flow_scaled"] = test_node_scaled[: ,0]
test_df["Area1_prod_scaled"] = test_node_scaled[: ,1]
test_df["Area2_prod_scaled"] = test_node_scaled[: ,2]
test_df["Area3_prod_scaled"] = test_node_scaled[: ,3]


node_attr_train = train_df[
    [
        "Area1-corr",
        "Area1-fault",
        "Area1-fault2",
        "Area1_prod_scaled",
        "inter_area_flow_scaled",
        "Area2-corr",
        "Area2-fault",
        "Area2-fault2",
        "Area2_prod_scaled",
        "inter_area_flow_scaled",
        "Area3-corr",
        "Area3-fault",
        "Area3-fault2",
        "Area3_prod_scaled",
        "inter_area_flow_scaled",
    ]
].values
node_attr_train = node_attr_train.reshape(int(len(node_attr_train)), 3, 5)

node_attr_test = test_df[
    [
        "Area1-corr",
        "Area1-fault",
        "Area1-fault2",
        "Area1_prod_scaled",
        "inter_area_flow_scaled",
        "Area2-corr",
        "Area2-fault",
        "Area2-fault2",
        "Area2_prod_scaled",
        "inter_area_flow_scaled",
        "Area3-corr",
        "Area3-fault",
        "Area3-fault2",
        "Area3_prod_scaled",
        "inter_area_flow_scaled",
    ]
].values
node_attr_test = node_attr_test.reshape(int(len(node_attr_test)), 3, 5)


# +
"""We start by creating a dataset. This is done piecewise by creating 
a datapoint per scenario. Each datapoint is a single graph.
The goal is to construct a list of datapoints and then create the datasets from those."""


"""
Map functions loop through all datapoints.
"""

training_set = list(map(div.create_dataset, y_train, edge_weights_train, node_attr_train))
test_set = list(map(div.create_dataset, y_test, edge_weights_test, node_attr_test))


"""
Set batch_size
"""

batch_size = 512
training_dataset = DataLoader(training_set, batch_size=batch_size)
test_dataset = DataLoader(test_set, batch_size=batch_size)


print(f"Number of training graphs: {len(training_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")


# +
data_example = next(
    iter(training_dataset)
)  # getting an example graph to get dimensions

"""
Here the model, optimizer function, loss-criterion and the
learning-rate scheduler is initalized. 

model : Initalizes the neuron network
optimizer : Initializes the backpropagation optimizer
"""

model = div.GCN(data_example, hidden_channels=12, dropout_rate=0.2, isthird_layer=False, is_classification=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)#75)


scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)  # Factor for learning rate decay

criterion = torch.nn.BCEWithLogitsLoss()


"""Below is a loop which trains the model over a set amount of epochs and 
which prints the current epoch and the RMSE values for the training-set and the test-set.
The RMSE values and the epoch values are stored in seperate lists.

num_epochs : Number of epochs in training
"""
num_epochs = 100
train_list = np.array([])
test_list = np.array([])
for epoch in range(1, num_epochs+1):
    div.train_model(model, training_dataset, optimizer, scheduler, criterion)
    train_acc = div.test_model(model, training_dataset, batch_size=batch_size, classification=True)
    test_acc = div.test_model(model, test_dataset, batch_size=batch_size, classification=True)
    current_acc_test = test_acc
    print(
        f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
    )
    train_list = np.concatenate((train_list, train_acc), axis=None)
    test_list = np.concatenate((test_list, test_acc), axis=None)



"""
Plot showing the difference between the target value and the predicted value
for the test set, target data in ascending order
"""
y_pred, y_target, x_values = div.plotter(model, test_dataset, is_classification=True)

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
Plot showing the change of accuracy for both the training set and 
the test set.
"""

epoch_list = np.arange(1, len(train_list)+1)
plt.plot(epoch_list, train_list, color="blue", label="Training set")
plt.plot(epoch_list, test_list, color="red", label="Test set")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc="lower left",
    ncol=2,
    mode="expand",
    borderaxespad=0.0,
)
plt.show()


# Calculate True and False Negatives and Positives

plot_df["TP"] = plot_df.apply(lambda row: row['target'] == row['pred'] == 0, axis=1)
plot_df["FN"] = plot_df.apply(lambda row: row['target'] == 0 != row['pred'], axis=1)

plot_df["TN"] = plot_df.apply(lambda row: row['target'] == row['pred'] == 1, axis=1)
plot_df["FP"] = plot_df.apply(lambda row: row['target'] == 1 != row['pred'], axis=1)

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(plot_df['target'], plot_df['pred']).ravel()
(tn, fp, fn, tp)
