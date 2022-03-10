import div_methods as div
import numpy as np
import pandas as pd
import random as random
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import DataLoader
import datetime

# +
"""
Estimating costs linked to failure scenarios, preventive and corrective strategies.

Import file with scenarios as a dataframe. Input data is results from the classification model.

prev_id: id of preventive strategy
Line10, Line11, Line18, Line19: Power flow in respective lines
Area1 prod, Area2 prod, Area3 prod: total production in respective areas
Area1 load, Area2 load, Area3 load: total load in respective areas
inter_area_flow: inter-area flow
fault: which line has fault in given scenario
corr: which area has corrective measure (load or production disconnected)
Total cost: Total cost of system operation for given scenario

Rest of columns are one-hot-encodings.
"""

"""
NB! If running script from Notebook, do not define df here.

NB: Commenting out this part, since the train and val sets are defined in classification have been used to train the model. Test sets are results from classification.
"""

#df = pd.read_csv(r'./results/input_regression_040322.csv')


# +
"""
train_test_split splits data into train and test while also shuffling. Train 70%, validation 20%, test 10%
random_state chooses seed for shuffling which gives more consistency for our process.

Splitting into train and test set while keeping all scenarios with the same prev_id in the same set
"""

#prev_ids = df.prev_id.unique()

#grouped = df.groupby(df.prev_id)

#df_list = []
#for prev_id in prev_ids:
#    df_list.append(grouped.get_group(prev_id))


#train_df_list, test_val_df_list = train_test_split(
#    df_list, test_size=0.3, random_state=1
#)
#val_df_list, test_df_list = train_test_split(
#    test_val_df_list, test_size=1 / 3, random_state=1
#)

#train_df = pd.concat(train_df_list)
#val_df = pd.concat(val_df_list)
#test_df = pd.concat(test_df_list)

# +
"""
Using absolute value for power flow, since were using MinMax-normalisation.
"""

edge_weights_train = abs(train_df[["Line10", "Line11", "Line18", "Line19"]].values)
edge_weights_val = abs(val_df[["Line10", "Line11", "Line18", "Line19"]].values)
edge_weights_test = abs(test_df[["Line10", "Line11", "Line18", "Line19"]].values)


cost_types = ["Dispatch costs", "Correction cost"]

y_train = train_df[cost_types].values
y_val = val_df[cost_types].values
y_test = test_df[cost_types].values

# +
""" 
Here, the user can choose which transformer to use by changing the name of variable transform_method. 
The transformation methods are:

PowerTransformer
QuantileTransformer
StandardScaler
MinMaxScaling (Normalization)

The training set is both fitted and transformed before it is reshaped back to its original form

Should maybe scale power flows in lines, load and production together since they're the same unit?
"""

transform_method = "MinMaxScaler"

scale_y = div.transform_data_method(transform_method)
scale_node = div.transform_data_method(transform_method)
scale_edge = div.transform_data_method(transform_method)


def create_input_data(
    y_dataset, edge_weights_dataset, dataset_df, is_training_set=False, batch_size=1
):
    """
    Created a function to avoid triple code. Can be moved to div_methods.py
    """
    # Scaling output arguments separately so they get the same importance in loss function
    # y_dataset = y_dataset.reshape((len(y_dataset) * 2, 1))
    node_attr_dataset = abs(
        dataset_df[["inter_area_flow", "Area1_prod", "Area2_prod", "Area3_prod"]].values
    )
    node_attr_dataset = node_attr_dataset.reshape((len(node_attr_dataset) * 4, 1))
    edge_weights_dataset = edge_weights_dataset.reshape(
        (len(edge_weights_dataset) * 4, 1)
    )

    if is_training_set:
        y_dataset = scale_y.fit_transform(y_dataset)
        node_attr_dataset = scale_node.fit_transform(node_attr_dataset)
        edge_weights_dataset = scale_edge.fit_transform(edge_weights_dataset)
    else:
        y_dataset = scale_y.transform(y_dataset)
        node_attr_dataset = scale_node.transform(node_attr_dataset)
        edge_weights_dataset = scale_edge.transform(edge_weights_dataset)

    #y_dataset = y_dataset.reshape(int(len(y_dataset) / 2), 2)
    node_attr_dataset = node_attr_dataset.reshape(int(len(node_attr_dataset) / 4), 4)
    edge_weights_dataset = edge_weights_dataset.reshape(
        int(len(edge_weights_dataset) / 4), 4
    )

    """
    Creating node attributes.
    """

    dataset_df.loc[:, "inter_area_flow_scaled"] = node_attr_dataset[:, 0]
    dataset_df.loc[:, "Area1_prod_scaled"] = node_attr_dataset[:, 1]
    dataset_df.loc[:, "Area2_prod_scaled"] = node_attr_dataset[:, 2]
    dataset_df.loc[:, "Area3_prod_scaled"] = node_attr_dataset[:, 3]

    node_scaled_dataset = dataset_df[
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
    node_scaled_dataset = node_scaled_dataset.reshape(
        int(len(node_scaled_dataset)), 3, 5
    )

    # Map function loops through all datapoints.
    graph_dataset = list(
        map(div.create_dataset, y_dataset, edge_weights_dataset, node_scaled_dataset)
    )

    return DataLoader(graph_dataset, batch_size=batch_size)


# +
"""
We start by creating a dataset. This is done piecewise by creating 
a datapoint per scenario. Each datapoint is a single graph.
The goal is to construct a list of data points and then create the datasets from those.

Possible to set batch size. So far, batch_size=1 has given best results.
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
learning-rate scheduler are initalized. 

model : Initalizes the neuron network
optimizer : Initializes the backpropagation optimizer
scheduler : Initalizes the learning-rate reducter function
criterion : Defines the error function
is_rmse : Boolean set to True if want RMSE as criterion. If false, MSE will be the criterion
is_third_layer: Boolean. If set to True, model will use 3 GCN layers. 
                Else, model will use 2 GCN-layers.
"""


is_rmse = True
is_third_layer = False

# Hyperparameters: Set based on hyperparameteroptimisation.
hidden_channels = 41
dropout_rate = 0.35
learning_rate = 0.0055

# Getting an example graph to get dimensions
data_example = next(iter(training_dataset))

model = div.GCN(
    data_example,
    hidden_channels=hidden_channels,
    output_dim=2,
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
The MSE/RMSE values and the epoch values are stored in separate lists.

Included simple early stopping to avoid overfitting. If the model has not improved over
n epochs, training is stopped.

num_epochs: Number of epochs in training
patience: number of epochs with no improvement before training is stopped
min_delta: minimum change qualified as improvement  

All these parameters are set based on visual inspection of convergence plot given data set

"""

num_epochs = 100

min_delta = 0.00001
patience = 10

train_list = np.array([])
val_list = np.array([])
err_diff = np.array([])
previous_error_val = 0

for epoch in range(1, num_epochs + 1):
    div.train_model(
        model, training_dataset, optimizer, scheduler, criterion, is_rmse=is_rmse, output_dim=2,
    )
    train_acc = div.test_model_regression(
        model, training_dataset, scale=scale_y, is_rmse=is_rmse, output_dim=2
    )
    val_acc = div.test_model_regression(
        model, val_dataset, scale=scale_y, is_rmse=is_rmse, output_dim=2
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
    if err_diff[-patience + 1:].mean() < min_delta:
        print(f"solution converged at epoch: {epoch}")
        break
    else:
        previous_error_val = current_error_val


# +
"""
df_results return results for test data after transforming back.
"""
test_acc = div.test_model_regression(
    model, test_dataset, scale=scale_y, is_rmse=is_rmse, output_dim=2)

y_pred, y_target, scenarios = div.plotter(model, test_dataset)

y_pred2 = scale_y.inverse_transform(np.array(y_pred).reshape(-1, 2))
y_target2 = scale_y.inverse_transform(np.array(y_target).reshape(-1, 2))

df_results = test_df.copy()
df_results[["Dispatch_cost_target", "Correction_cost_target"]] = y_target2
df_results[["Dispatch_cost_pred", "Correction_cost_pred"]] = y_pred2
# -

# +
"""Following is the hyperparameter-optimization loop done following the random search method.

The loop creates 50 models, each with a random learning rate, hidden channels number
and dropout rate number taken from a predefined range. 
The loop then starts to train the model over set number of epochs, or 
until the model has converged and the change in RMSE is lower than a set threshold.

The RMSE value and all the other hyperparameters are stored in each their separate list,
and when the loop has been iterated over, a second loop goes over the RMSE list and finds the
hyperparameter-settings with the lowest RMSE values.

is_tuning: Boolean. If true script runs hyperparameteroptimalisation.
"""

is_tuning = False

if is_tuning:
    begin_time = datetime.datetime.now()
    min_error_train_list = np.array([])
    min_error_val_list = np.array([])
    err_diff = np.array([])
    lr_list = np.array([])
    hidden_channels_list = np.array([])
    dropout_rate_list = np.array([])
    epoch_list = np.array([])

    min_delta = 0.000001
    patience = 10

    for i in range(50):
        print(f"Execution time for {i - 1}: {datetime.datetime.now() - begin_time}")
        lr = random.randrange(1, 101) / 10000  # [0.01, 0.0001]
        hidden_channels = random.randint(5, 51)
        dropout_rate = random.randrange(0, 10) / 20  # [0, 0.5]
        # is_third_layer = bool(random.getrandbits(1))
        print(f"Model n#: {i} with learning rate: {lr}, hidden channels: {hidden_channels} and dropout rate: {dropout_rate}.")
        num_epochs = 80

        model = div.GCN(
            data_example,
            hidden_channels=hidden_channels,
            output_dim=2,
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
                model, training_dataset, optimizer, scheduler, criterion, is_rmse=is_rmse, output_dim=2
            )
            train_acc = div.test_model_regression(
                model, training_dataset, scale=scale_y, is_rmse=is_rmse, output_dim=2
            )
            val_acc = div.test_model_regression(
                model, val_dataset, scale=scale_y, is_rmse=is_rmse, output_dim=2
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
            if err_diff[-patience + 1 :].mean() < min_delta:
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
    Here I find the index of the minimum value of the (R)MSE. This is done this way
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

