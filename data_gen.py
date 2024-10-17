from ANDA import anda
import torch
import pandas as pd
import os
import argparse

import config as cfg

# Get arguments
parser = argparse.ArgumentParser(description='Generate datasets for ANDA')
parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
args = parser.parse_args()

# valid dataset names
assert cfg.dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FMNIST', 'EMNIST'], \
        "Dataset not found! Please check the ANDA page for more details."

# Create folder if not exist
os.makedirs(f'./FedDrift/data/{cfg.dataset_name}', exist_ok=True)


anda_dataset = []

# special static mode using unique fn
if cfg.drifting_type == 'static':
    assert cfg.non_iid_type in ['feature_skew',
                                'label_skew',
                                'feature_condition_skew',
                                'label_condition_skew',
                                'split_unbalanced',
                                'feature_label_skew',
                                'feature_condition_skew_with_label_skew',
                                'label_condition_skew_with_label_skew',
                                'label_condition_skew_unbalanced',
                                'feature_condition_skew_unbalanced',
                                'label_skew_unbalanced',
                                'feature_skew_unbalanced',
                                'feature_skew_strict',
                                'label_skew_strict',
                                'feature_condition_skew_strict',
                                'label_condition_skew_strict',
    ], "Non-IID type not supported in static mode! Please check the ANDA page for more details."
    anda_dataset = anda.load_split_datasets(
        dataset_name = cfg.dataset_name,
        client_number = cfg.n_clients,
        non_iid_type = cfg.non_iid_type,
        mode = "manual",
        verbose = cfg.verbose,
        count_labels=cfg.count_labels,
        plot_clients=cfg.plot_clients,
        random_seed = cfg.random_seed + args.fold,
        **cfg.args
    )
elif cfg.drifting_type in ['trND_teDR','trDA_teDR','trDA_teND','trDR_teDR','trDR_teND']:
    # dynamic mode using same fn
    anda_dataset = anda.load_split_datasets_dynamic(
        dataset_name = cfg.dataset_name,
        client_number = cfg.n_clients,
        non_iid_type = cfg.non_iid_type,
        drfting_type = cfg.drifting_type,
        verbose = cfg.verbose,
        count_labels=cfg.count_labels,
        plot_clients=cfg.plot_clients,
        random_seed = cfg.random_seed + args.fold,
        **cfg.args
    )
else:
    raise ValueError("Drifting type not found! Please check the ANDA page for more details.")

# Save anda_dataset

# simple format as training not drifting
if not cfg.training_drifting:
    for client, client_data in enumerate(anda_dataset):
        # Get the training features and labels for this client
        train_features = client_data['train_features']
        train_labels = client_data['train_labels']
        test_features = client_data['test_features']
        test_labels = client_data['test_labels']
        
        # Split the training features and labels into 10 equal parts
        feature_pieces = torch.chunk(train_features, cfg.n_rounds)
        label_pieces = torch.chunk(train_labels, cfg.n_rounds)

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
                f'./FedDrift/data/{cfg.dataset_name}/' + 'client_{}_iter_{}.csv'.format(client, it),
                index=False
            )

            print(f"Saved: client_{client}_iter_{it}.csv")

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
            f'./FedDrift/data/{cfg.dataset_name}/' + 'client_{}_iter_{}.csv'.format(client, cfg.n_rounds),
            index=False
        )

        print(f"Saved: client_{client}_iter_{cfg.n_rounds}.csv")

# complex format as training drifting
else:
    drifting_log = {}
    for dataset in anda_dataset:
        client_number = dataset['client_number']
        cur_drifting_round = int(cfg.n_rounds * dataset['epoch_locker_indicator']) if dataset['epoch_locker_indicator'] != -1 else cfg.n_rounds
        # log drifting round info
        if client_number not in drifting_log:
            drifting_log[client_number] = []
        drifting_log[client_number].append(cur_drifting_round)

    for dataset in anda_dataset:
        client_number = dataset['client_number']
        cur_feature = dataset['features']
        cur_label = dataset['labels']

        f_flat = cur_feature.view(cur_feature.size(0), -1)  # Flatten each feature tensor (keeping batch size)
        l_flat = cur_label.view(cur_label.size(0), -1)  # Flatten each label tensor (keeping batch size)
        
        # Concatenate the flattened features and labels along the second dimension
        concat_data = torch.cat((f_flat, l_flat), dim=1)
        cur_data = concat_data.numpy()

        # Create appropriate headers
        num_features = f_flat.shape[1]
        num_labels = l_flat.shape[1]
        
        feature_headers = [f'f{i+1}' for i in range(num_features)]
        label_headers = ['label']
        headers = feature_headers + label_headers
        df = pd.DataFrame(cur_data, columns=headers)

        # find cur_drift training rounds
        cur_drifting_round = int(cfg.n_rounds * dataset['epoch_locker_indicator']) if dataset['epoch_locker_indicator'] != -1 else cfg.n_rounds

        cur_log = drifting_log.get(client_number, [])
        next_drift_round = min((r for r in cur_log if r > cur_drifting_round), default=cfg.n_rounds + 1)

        # temp
        # print(list(range(cur_drifting_round, next_drift_round)))

        for r in range(cur_drifting_round,next_drift_round):
            # Save the DataFrame to a CSV file
            df.to_csv(
                f'./FedDrift/data/{cfg.dataset_name}/' + 'client_{}_iter_{}.csv'.format(client_number, r),
                index=False
            )

            print(f"Saved: client_{client_number}_iter_{r}.csv")