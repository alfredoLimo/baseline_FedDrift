# 
#   Generate drift type ['static', 'trND_teDR'] datasets
#

from ANDA import anda
import torch
import pandas as pd
import os

# === copy-paste your config.py file here ===
random_seed = 42
n_clients = 10
dataset_name = "MNIST" # ["CIFAR10", "CIFAR100", "MNIST", "FMNIST", "EMNIST"]
drifting_type = 'static' # ['static', 'trND_teDR', 'trDA_teDR', 'trDA_teND', 'trDR_teDR', 'trDR_teND'] refer to ANDA page for more details
non_iid_type = 'label_skew' # refer to ANDA page for more details
verbose = True
count_labels = True
plot_clients = False
args = {
    # 'color_bank':3,
    # 'scaling_low':0.2,
    # 'scaling_high':0.5,
}
n_rounds = 10

os.makedirs(f'./FedDrift/data/{dataset_name}', exist_ok=True)

assert drifting_type in ['static', 'trND_teDR'], "Drift type not supported here"
assert dataset_name in ["CIFAR10", "CIFAR100", "MNIST", "FMNIST", "EMNIST"], "Dataset not supported here"

if drifting_type == 'static':
    anda_dataset = anda.load_split_datasets(
        dataset_name = dataset_name,
        client_number = n_clients,
        non_iid_type = non_iid_type,
        mode = "manual",
        verbose = verbose,
        count_labels = count_labels,
        plot_clients = plot_clients,
        random_seed = random_seed,
        **args
    )

else:
    anda_dataset = anda.load_split_datasets_dynamic(
        dataset_name = dataset_name,
        client_number = n_clients,
        non_iid_type = non_iid_type,
        drfting_type = 'trND_teDR',
        verbose = verbose,
        count_labels = count_labels,
        plot_clients = plot_clients,
        random_seed = random_seed,
        **args
    )

# Iterate over each client in the anda_dataset
for client, client_data in enumerate(anda_dataset):
    # Get the training features and labels for this client
    train_features = client_data['train_features']
    train_labels = client_data['train_labels']
    test_features = client_data['test_features']
    test_labels = client_data['test_labels']
    
    # Split the training features and labels into 10 equal parts
    feature_pieces = torch.chunk(train_features, n_rounds)
    label_pieces = torch.chunk(train_labels, n_rounds)

    # For each chunk, concatenate the flattened features and labels and save to CSV
    for it, (f, l) in enumerate(zip(feature_pieces, label_pieces)):
        # Flatten features and labels
        f_flat = f.view(f.size(0), -1)  # Flatten each feature tensor (keeping batch size)
        l_flat = l.view(l.size(0), -1)  # Flatten each label tensor (keeping batch size)
        
        # Concatenate the flattened features and labels along the second dimension
        concat_data = torch.cat((f_flat, l_flat), dim=1)

        # Convert the concatenated data to a numpy array
        train_data = concat_data.numpy()

        # Create appropriate headers
        num_features = f_flat.shape[1]
        num_labels = l_flat.shape[1]
        
        feature_headers = [f'f{i+1}' for i in range(num_features)]
        label_headers = ['label']
        headers = feature_headers + label_headers

        # Create a Pandas DataFrame with the headers
        df = pd.DataFrame(train_data, columns=headers)

        # Save the DataFrame to a CSV file
        df.to_csv(
            f'./FedDrift/data/{dataset_name}/' + 'client_{}_iter_{}.csv'.format(client, it),
            index=False
        )

        print(f"Saved: client_{client}_iter_{it}.csv with headers")

    # Save test features and labels without chunking
    f_flat = test_features.view(test_features.size(0), -1)  # Flatten test features
    l_flat = test_labels.view(test_labels.size(0), -1)  # Flatten test labels

    concat_test_data = torch.cat((f_flat, l_flat), dim=1)

    # Convert the concatenated test data to a numpy array
    test_data = concat_test_data.numpy()

    # Create headers for test data
    num_features = f_flat.shape[1]
    num_labels = l_flat.shape[1]

    feature_headers = [f'f{i+1}' for i in range(num_features)]
    label_headers = ['label']
    headers = feature_headers + label_headers

    # Create a Pandas DataFrame with the headers for test data
    df_test = pd.DataFrame(test_data, columns=headers)

    # Save the test DataFrame to a CSV file with iter number n_rounds
    df_test.to_csv(
        f'./FedDrift/data/{dataset_name}/' + 'client_{}_iter_{}.csv'.format(client, n_rounds),
        index=False
    )

    print(f"Saved: client_{client}_iter_{n_rounds}.csv with headers")

