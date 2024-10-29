#!/bin/bash

k_folds=$(python -c "from config import k_folds; print(k_folds)")
model_name=$(python -c "from config import model_name; print(model_name)")
dataset_name=$(python -c "from config import dataset_name; print(dataset_name)")
drifting_type=$(python -c "from config import drifting_type; print(drifting_type)")
non_iid_type=$(python -c "from config import non_iid_type; print(non_iid_type)")
n_clients=$(python -c "from config import n_clients; print(n_clients)")
n_rounds=$(python -c "from config import n_rounds; print(n_rounds)")
strategy=$(python -c "from config import strategy; print(strategy)")

CL_ALGO=$(python -c "from config import cl_algo; print(cl_algo)")
CL_ALGO_ARG=$(python -c "from config import cl_algo_arg; print(cl_algo_arg)")

echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
    MODEL: $model_name\033[0m\n\033[1;36m \
    Dataset: $dataset_name\033[0m\n\033[1;36m \
    Drifting type: $drifting_type\033[0m\n\033[1;36m \
    Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
    Number of clients: $n_clients\033[0m\n\033[1;36m \
    Number of rounds: $n_rounds\033[0m\n\033[0;31m \
    strategy: $strategy\033[0m\n\033[0;31m \
    baseline_CL_ALGO: $CL_ALGO\033[0m\n\033[0;31m \
    baseline_CL_ALGO_ARG: $CL_ALGO_ARG\033[0m\n\033[1;36m \
    K-Folds: $k_folds\033[0m\n"

# clean history
rm -rf output_[0-9]*.log

# K-Fold evaluation, if k_folds > 1
for fold in $(seq 0 $(($k_folds - 1))); do        
    echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

    # # Clean and create datasets
    # rm -rf ./FedDrift/data/MNIST/*.csv
    # rm -rf ./FedDrift/data/FMNIST/*.csv
    # rm -rf ./FedDrift/data/EMNIST/*.csv
    # rm -rf ./FedDrift/data/CIFAR10/*.csv
    # rm -rf ./FedDrift/data/CIFAR100/*.csv
    # python ./data_gen.py --fold "$fold"

    # go to dir
    cd FedDrift/fedml_experiments/distributed/fedavg_cont_ens

    ./run_fedavg_distributed_pytorch.sh "$fold" "$CL_ALGO" "$CL_ALGO_ARG"

    cd ../../../..
done

# Aggregate results
python average_results.py
