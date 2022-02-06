import div_methods as div
import numpy as np
import pandas as pd
import random as random
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import DataLoader

# +
"""
Estimating costs linked to failure scenarios, preventive and corrective strategies.

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

# df = pd.read_csv(r'data\tot_cost_failure_scenarios_041021_multicorr2.csv')
# df = pd.read_csv(r'./data/tot_cost_failure_scenarios_180122_01.csv')

# +
"""
If set_max_cost = True, values higher than max_cost will be removed
Or values higher than set_max can be set to set_max
"""

set_max_cost = True
max_cost = 1.128

if set_max_cost:
    df = df[df["Total cost"] < max_cost]
    # df.loc[df["Total cost"] > max_cost, "Total cost"] = max_cost


# +
"""
train_test_split splits data into train and test while also shuffling. Train 70%, validation 20%, test 10%
random_state chooses seed for shuffling which gives more consistency for our process.

When access to more data, should split into train, validation and test-set. ??

Splitting into train and test set while keeping all scenarios with the same prev_id in the same set
"""

prev_ids = df.prev_id.unique()

grouped = df.groupby(df.prev_id)

df_list = []
for prev_id in prev_ids:
    df_list.append(grouped.get_group(prev_id))


train_df_list, test_val_df_list = train_test_split(
    df_list, test_size=0.3, random_state=1
)
val_df_list, test_df_list = train_test_split(
    test_val_df_list, test_size=1 / 3, random_state=1
)

train_df = pd.concat(train_df_list)
val_df = pd.concat(val_df_list)
test_df = pd.concat(test_df_list)

# +
"""
Absolute value of power flow as edge weights, since were using undirected graphs.
"""

edge_weights_train = abs(train_df[["Line10", "Line11", "Line18", "Line19"]].values)
edge_weights_val = abs(val_df[["Line10", "Line11", "Line18", "Line19"]].values)
edge_weights_test = abs(test_df[["Line10", "Line11", "Line18", "Line19"]].values)

y_train = train_df["Total cost"].values
y_val = val_df["Total cost"].values
y_test = test_df["Total cost"].values

# +
""" 
Here, the user can choose which transformer to use by changing the name of variable transform_method. 
The transformation methods are:

PowerTransformer
QuantileTransformer
StandardScaler
MinMaxScaling (Normalization)

The training set is both fitted and transformed before it is reshaped back to its original form

Should maybe scale power flows in lines, load and production together since they're the same unit ??
"""

transform_method = "MinMaxScaler"

scale_y = div.transform_data_method(transform_method)
scale_node = div.transform_data_method(transform_method)
scale_edge = div.transform_data_method(transform_method)


def create_input_data(
    y_dataset, edge_weights_dataset, dataset_df, is_training_set=False, batch_size=1
):
    """
    Created a function to avoid triple code.
    """

    y_dataset = y_dataset.reshape((len(y_dataset), 1))
    node_attr_dataset = abs(
        dataset_df[["inter_area_flow", "Area1_prod", "Area2_prod", "Area3_prod"]].values
    )
    node_attr_dataset = node_attr_dataset.reshape((len(node_attr_dataset) * 4, 1))
    edge_weights_dataset = edge_weights_dataset.reshape(
        (len(edge_weights_dataset) * 4, 1)
    )

    if is_training_set:
        y_dataset = scale_y.fit_transform(y_dataset)
        node_scaled_dataset = scale_node.fit_transform(
            dataset_df[["inter_area_flow", "Area1_prod", "Area2_prod", "Area3_prod"]]
        )
        edge_weights_dataset = scale_edge.fit_transform(edge_weights_dataset)
    else:
        y_dataset = scale_y.transform(y_dataset)
        node_scaled_dataset = scale_node.transform(
            dataset_df[["inter_area_flow", "Area1_prod", "Area2_prod", "Area3_prod"]]
        )
        edge_weights_dataset = scale_edge.transform(edge_weights_dataset)

    y_dataset = y_dataset.reshape(-1)
    node_scaled_dataset = node_attr_dataset.reshape(int(len(node_attr_dataset) / 4), 4)
    edge_weights_dataset = edge_weights_dataset.reshape(
        int(len(edge_weights_dataset) / 4), 4
    )

    """
    Creating node attributes.
    """

    dataset_df.loc[:, "inter_area_flow_scaled"] = node_scaled_dataset[:, 0]
    dataset_df.loc[:, "Area1_prod_scaled"] = node_scaled_dataset[:, 1]
    dataset_df.loc[:, "Area2_prod_scaled"] = node_scaled_dataset[:, 2]
    dataset_df.loc[:, "Area3_prod_scaled"] = node_scaled_dataset[:, 3]

    node_attr_dataset = dataset_df[
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
    node_attr_dataset = node_attr_dataset.reshape(int(len(node_attr_dataset)), 3, 5)

    # Map function loops through all datapoints.
    graph_dataset = list(
        map(div.create_dataset, y_dataset, edge_weights_dataset, node_attr_dataset)
    )

    return DataLoader(graph_dataset, batch_size=batch_size)


# +
"""
We start by creating a dataset. This is done piecewise by creating 
a datapoint per scenario. Each datapoint is a single graph.
The goal is to construct a list of datapoints and then create the datasets from those.

Possible to set batch size. So far, batch_size=1 given best results.
"""

batch_size = 1

training_dataset = create_input_data(
    y_train, edge_weights_train, train_df, is_training_set=True, batch_size=batch_size
)
val_dataset = create_input_data(
    y_val, edge_weights_val, val_df, is_training_set=False, batch_size=batch_size
)
test_dataset = create_input_data(
    y_test, edge_weights_test, test_df, is_training_set=False, batch_size=batch_size
)

print(f"Number of training graphs: {len(training_dataset)}")
print(f"Number of validation graphs: {len(val_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")


# +
"""
Here the model, optimizer function, loss-criterion and the
learning-rate scheduler is initalized. 

model : Initalizes the neuron network
optimizer : Initializes the backpropagation optimizer
scheduler : Initalizes the learning-rate reducter function
criterion : Defines the error function
is_rmse : Boolean set to True if want RMSE as criterion. If false, MSE will be the criterion
is_third_layer: Boolean. If set to True, model will use 3 GCN layers. 
                Else, model will use 2 GCN-layers.
"""


is_rmse = False
is_third_layer = True

# Hyperparameters: Set based on hyperparameteroptimisation.
hidden_channels = 12
dropout_rate = 0.2
learning_rate = 0.0075

# Getting an example graph to get dimensions
data_example = next(iter(training_dataset))


model = div.GCN(
    data_example,
    hidden_channels=hidden_channels,
    dropout_rate=dropout_rate,
    isthird_layer=is_third_layer,
)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)  # Factor for learning rate decay

criterion = torch.nn.MSELoss()

# +
"""
Below is a loop which trains the model over a set amount of epochs and 
which prints the current epoch and the RMSE/MSE values for the training-set and the test-set.
The MSE/RMSE values and the epoch values are stored in seperate lists.
In addition, a breakpoint (patience_value) has been defined so that if the mean MSE/RMSE is less than a 
desired value for n number of epochs (patience_num), the training stops to prevent overfitting.

num_epochs : Number of epochs in training
"""

num_epochs = 80

patience_value = 0.00000001
patience_num = 5

train_list = np.array([])
val_list = np.array([])
err_diff = np.array([])
previous_error_val = 0

for epoch in range(1, num_epochs + 1):
    div.train_model(
        model, training_dataset, optimizer, scheduler, criterion, calc_rmse=is_rmse
    )
    train_acc = div.test_model(
        model, training_dataset, batch_size=batch_size, scale=scale_y, calc_rmse=is_rmse
    )
    val_acc = div.test_model(
        model, val_dataset, batch_size=batch_size, scale=scale_y, calc_rmse=is_rmse
    )
    current_error_val = val_acc
    print(
        f"Epoch: {epoch:03d}, Train Acc: {train_acc:.6f}, Validation Acc: {val_acc:.6f}"
    )
    train_list = np.concatenate((train_list, train_acc), axis=None)
    val_list = np.concatenate((val_list, val_acc), axis=None)
    err_diff = np.concatenate(
        (err_diff, abs(previous_error_val - current_error_val)), axis=None
    )
    if err_diff[-patience_num + 1 :].mean() < patience_value:
        print(f"solution converged at epoch: {epoch}")
        break
    else:
        previous_error_val = current_error_val


# +
"""
df_results return results for test data
"""
test_acc = div.test_model(
    model, test_dataset, batch_size=batch_size, scale=scale_y, calc_rmse=is_rmse
)
y_pred, y_prob, y_target, x_values = div.plotter(model, test_dataset)

y_pred = scale_y.inverse_transform(np.array(y_pred).reshape(1, -1))
y_target = scale_y.inverse_transform(np.array(y_target).reshape(1, -1))

df_results = test_df.copy()
df_results["target"] = y_target[0]
df_results["pred"] = y_pred[0]
# -


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
min_error_train_list = np.array([])
min_error_val_list = np.array([])
err_diff = np.array([])
lr_list = np.array([])
hidden_channels_list = np.array([])
dropout_rate_list = np.array([])
epoch_list = np.array([])

patience_value = 0.000001
patience_num = 10

for i in range(50):
    lr = random.randrange(1, 101) / 10000  # [0.01, 0.0001]
    hidden_channels = random.randint(5, 51)
    dropout_rate = random.randrange(0, 10) / 20  # [0, 0.5]
    # is_third_layer = bool(random.getrandbits(1))
    num_epochs = 80

    model = div.GCN(
        data_example,
        hidden_channels=hidden_channels,
        dropout_rate=dropout_rate,
        isthird_layer=is_third_layer,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    min_error_train = 0
    min_error_val = 0
    previous_error_val = 0
    for epoch in range(1, num_epochs + 1):
        div.train_model(
            model, training_dataset, optimizer, scheduler, criterion, calc_rmse=is_rmse
        )
        train_acc = div.test_model(
            model,
            training_dataset,
            batch_size=batch_size,
            scale=scale_y,
            calc_rmse=is_rmse,
        )
        val_acc = div.test_model(
            model, val_dataset, batch_size=batch_size, scale=scale_y, calc_rmse=is_rmse
        )
        current_error_train = train_acc
        current_error_val = val_acc
        """Here the min values are initialized if this is the first time through the loop.
        """
        if epoch == 1:
            min_error_train = current_error_train
            min_error_val = current_error_val
        """Here the minimum value for the train & validation rmse is changed if the current value is smaller than the current min value
        """
        if current_error_train < min_error_train:
            min_error_train = current_error_train

        if current_error_val < min_error_val:
            min_error_val = current_error_val
        """Here I create the option to break out of the loop if the solution converges
        """
        err_diff = np.concatenate(
            (err_diff, abs(previous_error_val - current_error_val)), axis=None
        )
        if err_diff[-patience_num + 1 :].mean() < patience_value:
            print(f"solution converged at epoch: {epoch}")
            break
        else:
            previous_error_val = current_error_val

    min_error_train_list = np.concatenate(
        (min_error_train_list, min_error_train), axis=None
    )
    min_error_val_list = np.concatenate((min_error_val_list, min_error_val), axis=None)
    lr_list = np.append(lr_list, lr)
    hidden_channels_list = np.append(hidden_channels_list, hidden_channels)
    dropout_rate_list = np.append(dropout_rate_list, dropout_rate)
    epoch_list = np.append(epoch_list, epoch)


# +
"""
Here I find the index of the minimum value of the RMSE. This is done this way
as pytorch don't have a method of doing it.
"""

value = 0
min_error = 0
for i in range(len(min_error_val_list)):
    if min_error == 0:
        min_error = min_error_val_list[i]

    if min_error_val_list[i] < min_error:
        min_error = min_error_val_list[i]
        value = i

print(min_error)
print(value)

print(
    f"The lowest total MSE value, {min_error_val_list[value]:.7f}, for the test set was found with the following settings: "
)
print(
    f"a learning rate of: {lr_list[value]:.4f}, a hidden channel nr of: {hidden_channels_list[value]}"
)
print(
    f"a dropout rate of: {dropout_rate_list[value]:.4f} and an epoch number of: {epoch_list[value]}"
)
