import logging
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import torchvision

from fedml_api.distributed.fedavg.utils import transform_tensor_to_list, transform_list_to_tensor

class FedAvgEnsTrainerSoftCluster(object):
    def __init__(self, client_index, train_data_local_dicts, train_data_local_num_dicts, train_data_nums, all_local_data, device, models,
                 args):
        self.client_index = client_index
        self.train_data_local_dicts = train_data_local_dicts
        self.train_data_local_num_dicts = train_data_local_num_dicts
        self.all_train_data_nums = train_data_nums
        self.all_local_data = all_local_data

        self.device = device
        # print(f"Device: {self.device}", flush=True)
        self.args = args
        self.models = models
        # logging.info(self.model)        
        self.optimizers = []
        self.criterions = []
        for m in self.models:
            # self._freeze_layers(m)
            #m.to(self.device)  #Move model to GPU one at a time to save GPU memory
            self.criterions.append(nn.CrossEntropyLoss().to(self.device))
            if self.args.client_optimizer == "sgd":
                self.optimizers.append(torch.optim.SGD(m.parameters(), lr=self.args.lr, momentum=0.9))
            else:
                # self.optimizers.append(torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()),
                #                                         lr=self.args.lr,
                #                                         weight_decay=self.args.wd, amsgrad=True))   # MODIFIED: we only use SGD
                self.optimizers.append(torch.optim.SGD(m.parameters(), lr=self.args.lr, momentum=0.9))

    
    def _freeze_layers(self, model):
        if isinstance(model, torchvision.models.densenet.DenseNet):
            for name, param in model.named_parameters():
                param.requires_grad = self._is_trainable_layer(name)
    
    def _is_trainable_layer(self, layer_name):
        valid_substrings = ['classifier']
        valid_substrings += ['norm5', 'denseblock4']
        for s in valid_substrings:
            if s in layer_name:
                return True
        return False
    
    def update_model(self, weights, extra_info):
        # logging.info("update_model. client_index = %d" % self.client_index)
        for m, w in zip(self.models, weights):
            if self.args.is_mobile == 1:
                w = transform_list_to_tensor(w)

            m.load_state_dict(w)
        self.extra_info = extra_info

    def update_dataset(self, client_index):
        self.client_index = client_index

    def train(self):
        results = {}

        for mod_idx, model in enumerate(self.models):
            # For efficiency, don't train a model that is not being used at the current iteration
            if not any( self.extra_info['sc_weights'][self.args.curr_train_iteration][mod_idx] ):
                results[mod_idx] = (None, 0)
                continue
            
            unnorm_probs = np.asarray([self.extra_info['sc_weights'][t][mod_idx][self.client_index] * len(self.all_local_data[t])
                                       for t in range(len(self.all_local_data))]) 
            local_sample_number = sum(unnorm_probs)
            
            # Skip the training if all training data are weighted 0
            if local_sample_number == 0:
                results[mod_idx] = (None, 0)
                continue
                
            probs = unnorm_probs/sum(unnorm_probs)
            
            model.to(self.device)
            # change to train mode
            model.train()
            criterion = self.criterions[mod_idx]
            optimizer = self.optimizers[mod_idx]
            
            # self._freeze_layers(model)

            # NEW CODE FOR TRAINING
            dataset_path = f"./../../../data/{self.args.dataset}/client_{self.client_index}_train.csv"
            cur_data = pd.read_csv(dataset_path)
            # print(f"Data shape: {cur_data.shape}", flush=True)
            # Separate features (all columns except the last one)
            cur_features = cur_data.iloc[:, :-1]
            cur_labels = cur_data.iloc[:, -1]
            # Calculate the index for 80% of the data
            cur_features_train, _, cur_labels_train, _ = train_test_split(
                cur_features, cur_labels, train_size=0.8, random_state=self.args.seed
            )
            
            # reduce client data
            if self.args.n_samples_clients > 0:
                cur_features_train = cur_features_train[:self.args.n_samples_clients]
                cur_labels_train = cur_labels_train[:self.args.n_samples_clients]
            
            batch_size = self.args.batch_size

            # Convert features and labels to PyTorch tensors
            cur_features_tensor = torch.tensor(cur_features_train.values, dtype=torch.float32)
            cur_labels_tensor = torch.tensor(cur_labels_train.values, dtype=torch.long)  # Assuming classification task

            # Create a TensorDataset and DataLoader for batching
            dataset = TensorDataset(cur_features_tensor, cur_labels_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training loop
            for step in range(self.args.epochs):
                for x, labels in dataloader:  # Use dataloader for batches
                    x, labels = x.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()

                # if all( isinstance(self.all_local_data[t], list) for t in range(len(self.all_local_data)) ):
                #     batches = []

                #     for t in range(len(self.all_local_data)):
                #         if unnorm_probs[t] > 0:
                #             batches += self.all_local_data[t]
                
                #     for step in range(self.args.epochs):
                #         # # general case: fractional weights
                #         # t_sample = np.random.choice(len(self.all_local_data), p=probs)
                #         # data_t = self.all_local_data[t_sample]
                #         # batch_idx = np.random.choice(len(data_t))
                #         # (x, labels) = data_t[batch_idx]
                        
                #         # special case: weights are unit
                #         for x, labels in batches:
                #             x, labels = x.to(self.device), labels.to(self.device)
                #             optimizer.zero_grad()
                #             log_probs = model(x)
                #             loss = criterion(log_probs, labels)
                #             loss.backward()
                #             optimizer.step()

                #         # batch_idx = np.random.choice(len(batches))
                #         # (x, labels) = batches[batch_idx]
                #         # print(f"Xdim: {x.shape}", flush=True)   
                        
                #         # x, labels = x.to(self.device), labels.to(self.device)
                #         # optimizer.zero_grad()
                #         # log_probs = model(x)
                #         # loss = criterion(log_probs, labels)
                #         # loss.backward()
                #         # optimizer.step()
                # else:
                #     raise NotImplementedError("Only list of batches is supported for now")
                #         # t_sample = np.random.choice(len(self.all_local_data), p=probs)
                #         # data_t = self.all_local_data[t_sample]
                #         # (x, labels) = next(iter(data_t))
                #         # print(f"Xdim - 2: {x.shape}", flush=True)   
                        
                #         # x, labels = x.to(self.device), labels.to(self.device)
                #         # optimizer.zero_grad()
                #         # log_probs = model(x)
                #         # loss = criterion(log_probs, labels)
                #         # loss.backward()
                #         # optimizer.step()

            model.to(torch.device('cpu'))
            weights = model.state_dict()

            # transform Tensor to list
            if self.args.is_mobile == 1:
                weights = transform_tensor_to_list(weights)
            results[mod_idx] = (weights, local_sample_number)
            
        return results
