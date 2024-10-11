#!/bin/bash

# datasets
# rm -rf ./FedDrift/data/MNIST/*.csv
# rm -rf ./FedDrift/data/FMNIST/*.csv
# rm -rf ./FedDrift/data/EMNIST/*.csv
# rm -rf ./FedDrift/data/CIFAR10/*.csv
# rm -rf ./FedDrift/data/CIFAR100/*.csv

# # Change your settings here
# python ./datagen_train_static.py

wandb login

cd FedDrift/fedml_experiments/distributed/fedavg_cont_ens

# Change your settings here
./run_fedavg_distributed_pytorch.sh