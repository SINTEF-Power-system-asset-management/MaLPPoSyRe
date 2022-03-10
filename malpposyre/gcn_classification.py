import div_methods as div
import numpy as np
import pandas as pd
import random as random
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import DataLoader

# +
"""
Classifying costs linked to failure scenarios, preventive and corrective strategies.
data.

Import file with scenarios as a dataframe.

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

# df = pd.read_csv(r'./data/tot_cost_failure_scenarios_240222.csv')

# +
"""
Classify scenario as high or low cost based on threshold.
"""

# thresh = 1.13

df["Class"] = df.apply(lambda row: 0.0 if row["Total cost"] < thresh else 1.0, axis=1)

# +

"""
train_test_split splits data into train and test while also shuffling. Train 70%, validation 10%, test 20%
random_state chooses seed for shuffling which gives more consistency for our process.

Test dataset need to be large since this is what is used in regression model.

Splitting into train and test set while keeping all scenarios with the same prev_id in the same set.
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
Absolute value of power flow as edge weights, since were using MinMax-normalisation.
"""

edge_weights_train = abs(train_df[["Line10", "Line11", "Line18", "Line19"]].values)
edge_weights_val = abs(val_df[["Line10", "Line11", "Line18", "Line19"]].values)
edge_weights_test = abs(test_df[["Line10", "Line11", "Line18", "Line19"]].values)

y_train = train_df["Class"].values
y_val = val_df["Class"].values
y_test = test_df["Class"].values


# +
""" 
Here, the user can choose which transformer to use by changing the name of variable transform_method. 
The transformation methods are:

PowerTransformer
QuantileTransformer
StandardScaler
MinMaxScaling (Normalization)
"""

transform_method = "MinMaxScaler"

scale_y = div.transform_data_method(transform_method)
scale_node = div.transform_data_method(transform_method)
scale_edge = div.transform_data_method(transform_method)


def create_input_data(
    y_dataset, edge_weights_dataset, dataset_df, is_training_set=False, batch_size=1
):
    """
    Can be generalised and moved to div_methods.py. Similar function used in other files.
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
        node_attr_dataset = scale_node.fit_transform(node_attr_dataset)
        edge_weights_dataset = scale_edge.fit_transform(edge_weights_dataset)
    else:
        y_dataset = scale_y.transform(y_dataset)
        node_attr_dataset = scale_node.transform(node_attr_dataset)
        edge_weights_dataset = scale_edge.transform(edge_weights_dataset)

    y_dataset = y_dataset.reshape(-1)
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


"""
Possible to set batch size. The batch size defines the number of 
samples that will be propagated through the network.
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
is_third_layer: Boolean. If set to True, model will use 3 GCN layers. 
                Else, model will use 2 GCN-layers.
"""

is_third_layer = False

# Hyperparameters: Set based on hyperparameteroptimisation.
hidden_channels = 12
dropout_rate = 0.2
learning_rate = 0.0005

# Getting an example graph to get dimensions
data_example = next(
    iter(training_dataset)
)

model = div.GCN(
    data_example,
    hidden_channels=hidden_channels,
    dropout_rate=dropout_rate,
    isthird_layer=is_third_layer,
    is_classification=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95
)  # Factor for learning rate decay

# Weight based on the ratio between the majority and the minority class in training dataset
pos_weight = 1/((y_train.sum())/len(y_train))

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

"""
Possible to set a thresh probability which is used as input for test_model_classification
This is used for training. 

On the test set, thresh probability is set in the Notebook based on F2-score.
"""

thresh_prob = 0.28

# +
"""
Below is a loop which trains the model over a set amount of epochs and 
which prints the current epoch and the accuracy for the training-set and the test-set.
Included simple early stopping to avoid overfitting. If the model has not improved over
n epochs, training is stopped.

num_epochs: Number of epochs in training
patience: number of epochs with no improvement before training is stopped
min_delta: minimum change qualified as improvement  

All these parameters are set based on visual inspection of convergence plot given data set
"""

num_epochs = 30

min_delta = 0.001
patience = 10

train_list = np.array([])
val_list = np.array([])
acc_diff = np.array([])
previous_acc_val = 0

for epoch in range(1, num_epochs + 1):
    div.train_model(model, training_dataset, optimizer, scheduler, criterion)
    train_acc = div.test_model_classification(model, training_dataset, thresh_prob=thresh_prob)
    val_acc = div.test_model_classification(model, val_dataset, thresh_prob=thresh_prob)
    current_acc_val = val_acc
    print(
        f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}"
    )
    train_list = np.concatenate((train_list, train_acc), axis=None)
    val_list = np.concatenate((val_list, val_acc), axis=None)
    acc_diff = np.concatenate(
        (acc_diff, abs(previous_acc_val - current_acc_val)), axis=None
    )
    if acc_diff[-patience + 1 :].mean() < min_delta:
        print(f"solution converged at epoch: {epoch}")
        break
    else:
        previous_acc_val = current_acc_val


# +
"""
df_results_classification return results for test data
"""

y_prob, y_target, x_values = div.plotter(model, test_dataset)

df_results = test_df.copy()
df_results["target"] = y_target
df_results["prob"] = y_prob


# +
"""Following is the hyperparameter-optimization loop done following the random search method.

The loop creates 50 models, each with a random learning rate, hidden channels number
and dropout rate number taken from a predefined range. 
The loop then starts to train the model over chosen number of epochs, or 
until the model has converged and the change in accuracy is lower than a set threshold.

The accuracy value and all the other hyperparameters are stored in each their separate list,
and when the loop has been iterated over, a second loop goes over the accuracy list and finds the
hyperparameter-settings with the lowest accuracy values.
"""

is_tuning = False

if is_tuning:
    max_acc_train_list = np.array([])
    max_acc_val_list = np.array([])
    lr_list = np.array([])
    hidden_channels_list = np.array([])
    dropout_rate_list = np.array([])
    epoch_list = np.array([])

    for i in range(50):
        lr = random.randrange(1, 101) / 100000  # [0.001, 0.00001]
        hidden_channels = random.randint(5, 51)
        dropout_rate = random.randrange(0, 10) / 20  # [0, 0.5]
        # is_third_layer = bool(random.getrandbits(1))
        num_epochs = 50

        model = div.GCN(
            data_example,
            hidden_channels=hidden_channels,
            dropout_rate=dropout_rate,
            isthird_layer=is_third_layer,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        previous_acc_val = 0
        for epoch in range(1, num_epochs + 1):
            div.train_model(model, training_dataset, optimizer, scheduler, criterion)
            train_acc = div.test_model(
                model, training_dataset, batch_size=batch_size, classification=True
            )
            val_acc = div.test_model(
                model, val_dataset, batch_size=batch_size, classification=True
            )
            current_acc_train = train_acc
            current_acc_val = val_acc
            """Here the min values are initialized if this is the first time through the loop.
            """
            if epoch == 1:
                max_acc_train = current_acc_train
                max_acc_val = current_acc_val
            """Here the minimum value for the train & val rmse is changed if the current value is smaller than the current min value
            """
            if current_acc_train > max_acc_train:
                max_acc_train = current_acc_train

            if current_acc_val > max_acc_val:
                max_acc_val = current_acc_val
            """Here I create the option to break out of the loop if the solution converges
            """
            acc_diff = np.concatenate(
                (acc_diff, abs(previous_acc_val - current_acc_val)), axis=None
            )
            if acc_diff[-patience + 1 :].mean() < min_delta:
                print(f"solution converged at epoch: {epoch}")
                break
            else:
                previous_acc_val = current_acc_val

        max_acc_train_list = np.concatenate((max_acc_train_list, max_acc_train), axis=None)
        max_acc_val_list = np.concatenate((max_acc_val_list, max_acc_val), axis=None)
        lr_list = np.append(lr_list, lr)
        hidden_channels_list =np.append(hidden_channels_list, hidden_channels)
        dropout_rate_list = np.append(dropout_rate_list, dropout_rate)
        epoch_list = np.append(epoch_list, epoch)


    # +
    """
    Here I find the index of the max accuracy. This is done this way
    as pytorch doesn't have a method of doing it.
    """

    value = 0
    for i in range(len(max_acc_val_list)):
        if i == 0:
            max_acc = max_acc_val_list[i]

        if max_acc_val_list[i] > max_acc:
            max_acc = max_acc_val_list[i]
            value = i

    print(max_acc)
    print(value)

    print(
        f"The lowest total MSE value, {max_acc_val_list[value]:.7f}, for the vallidation set was found with the following settings: "
    )
    print(
        f"a learning rate of: {lr_list[value]:.4f}, a hidden channel nr of: {hidden_channels_list[value]}"
    )
    print(
        f"a dropout rate of: {dropout_rate_list[value]:.4f} and an epoch number of: {epoch_list[value]}"
    )

