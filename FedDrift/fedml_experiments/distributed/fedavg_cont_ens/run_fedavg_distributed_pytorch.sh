#!/bin/bash
CLIENT_NUM=10
WORKER_NUM=10
SERVER_NUM=1
GPU_NUM_PER_SERVER=4
MODEL=fnn
DISTRIBUTION=homo
ROUND=10
EPOCH=2
BATCH_SIZE=64
LR=0.001
DATASET=MNIST
DATA_DIR="./../../../data/"
SAMPLE_NUM=100
NOISE_PROB=0
CI=0
TRAIN_ITER=10
CONCEPT_NUM=4
RESET_MODELS=0
DRIFT_TOGETHER=0
CL_ALGO=softcluster
CL_ALGO_ARG=H_A_C_1_10_0
TIME_STRETCH=1
DUMMY_ARG=0
CHANGE_POINTS=A

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

# First we prepare the data for multiple training iterations
# TODO: handle the case for multiple nodes

# python3 ./prepare_data.py \
#   --dataset $DATASET \
#   --data_dir $DATA_DIR \
#   --sample_num $SAMPLE_NUM \
#   --noise_prob $NOISE_PROB \
#   --partition_method $DISTRIBUTION \
#   --client_num_in_total $CLIENT_NUM \
#   --client_num_per_round $WORKER_NUM \
#   --batch_size $BATCH_SIZE \
#   --train_iteration $TRAIN_ITER \
#   --drift_together $DRIFT_TOGETHER \
#   --time_stretch $TIME_STRETCH \
#   --change_points "${CHANGE_POINTS:-rand}"

# Execute the training for one iteration at a time
# We do this because the FedML framework calls MPI_Abort whenever
# the training reaches the target round, and changing the
# framework to handle multiple training iterations would break
# most of the existing codes.

TI=${TRAIN_ITER}
for (( it=0; it < TI; it++ ));
do
    mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
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
           --dummy_arg $DUMMY_ARG \
           --change_points "${CHANGE_POINTS:-rand}"
done

