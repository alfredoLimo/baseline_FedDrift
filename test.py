from ANDA import anda
import torch
import pandas as pd
import os

anda_dataset = anda.load_split_datasets(
    dataset_name = "MNIST",
    client_number = 10,
    non_iid_type = "label_skew",
    mode = "auto",
    non_iid_level = "low",
    verbose = False,
    count_labels = False,
    plot_clients = False,
    random_seed = 42
)

# Iterate over each client in the anda_dataset
for c, client_data in enumerate(anda_dataset):
    # Get the training features and labels for this client
    train_features = client_data['train_features']
    train_labels = client_data['train_labels']
    
    # Split the training features and labels into 10 equal parts
    feature_pieces = torch.chunk(train_features, 11)
    label_pieces = torch.chunk(train_labels, 11)

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
            "./temp_data/" + 'client_{}_iter_{}.csv'.format(c, it),
            index=False
        )

        print(f"Saved: client_{c}_iter_{it}.csv with headers")
