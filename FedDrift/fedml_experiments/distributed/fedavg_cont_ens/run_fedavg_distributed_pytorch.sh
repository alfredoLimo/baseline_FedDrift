#!/bin/bash
fold=$1
CL_ALGO=$2   
CL_ALGO_ARG=$3
CLIENT_NUM=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import n_clients; print(n_clients)")
WORKER_NUM=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import n_clients; print(n_clients)")
SEED=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import random_seed; print(random_seed)")
MODEL=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import model_name; print(model_name)")
EPOCH=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import local_epochs; print(local_epochs)")
BATCH_SIZE=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import batch_size; print(batch_size)")
LR=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import lr; print(lr)")
DATASET=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import dataset_name; print(dataset_name)")
TRAIN_ITER=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import n_rounds; print(n_rounds)")
N_SAMPLES_CLIENTS=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import n_samples_clients; print(n_samples_clients)")

CONCEPT_NUM=$(PYTHONPATH=$(realpath ./../../../../) python -c "from config import n_clients; print(n_clients)")




ROUND=1            
TIME_STRETCH=1
SERVER_NUM=1                 
CHANGE_POINTS=A
GPU_NUM_PER_SERVER=2
DATA_DIR="./../../../data/"
NOISE_PROB=0
CI=0
RESET_MODELS=0      
DRIFT_TOGETHER=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
# echo $PROCESS_NUM

hostname > mpi_host_file

TI=${TRAIN_ITER}
for (( it=0; it < TI; it++ ));
do
    taskset -c 0-80 mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
           --gpu_server_num $SERVER_NUM \
           --gpu_num_per_server $GPU_NUM_PER_SERVER \
           --model $MODEL \
           --dataset $DATASET \
           --data_dir $DATA_DIR \
           --noise_prob $NOISE_PROB \
           --client_num_in_total $CLIENT_NUM \
           --client_num_per_round $WORKER_NUM \
           --comm_round $ROUND \
           --epochs $EPOCH \
           --batch_size $BATCH_SIZE \
           --lr $LR \
           --ci $CI \
           --total_train_iteration $TI \
           --curr_train_iteration $it \
           --concept_num $CONCEPT_NUM \
           --reset_models $RESET_MODELS \
           --drift_together $DRIFT_TOGETHER \
           --report_client 1 \
           --retrain_data win-1 \
           --concept_drift_algo $CL_ALGO \
           --concept_drift_algo_arg $CL_ALGO_ARG \
           --time_stretch $TIME_STRETCH \
           --seed $SEED \
           --change_points "${CHANGE_POINTS:-rand}" \
           --fold $fold \
           --n_samples_clients $N_SAMPLES_CLIENTS
done
