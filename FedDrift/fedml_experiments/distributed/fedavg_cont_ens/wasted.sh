#!/bin/bash
CLIENT_NUM=10               # checked
SERVER_NUM=1                # checked
MODEL=fnn                   # TODO
ROUND=5                     # ?
EPOCH=5                     # checked
BATCH_SIZE=500              # checked
LR=0.01                     # checked
DATASET=MNIST               # TODO
TRAIN_ITER=10               # checked
CONCEPT_NUM=4               # ? prior knowledge?
CL_ALGO=softcluster         # BASELINES
CL_ALGO_ARG=H_A_C_1_10_0    # BASELINES
SEED=42                     # checked

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

CHANGE_POINTS=A
PROCESS_NUM=`expr $WORKER_NUM + 1`
hostname > mpi_host_file

for (( it=0; it < TRAIN_ITER; it++ ));
do
    mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
           --client_num_in_total $CLIENT_NUM \
           --client_num_per_round $CLIENT_NUM \
           --gpu_server_num $SERVER_NUM \
           --gpu_num_per_server 4 \
           --model $MODEL \
           --comm_round $ROUND \
           --epochs $EPOCH \
           --batch_size $BATCH_SIZE \
           --lr $LR \
           --dataset $DATASET \
           --data_dir "./../../../data/" \
           --noise_prob 0 \
           --ci 0 \
           --total_train_iteration $TRAIN_ITER \
           --concept_num $CONCEPT_NUM \
           --reset_models 0 \
           --drift_together 0 \
           --concept_drift_algo $CL_ALGO \
           --concept_drift_algo_arg $CL_ALGO_ARG \
           --time_stretch 1 \
           --seed $SEED \
           --change_points "${CHANGE_POINTS:-rand}" \
           --curr_train_iteration $it \
           --report_client 1 \
           --retrain_data win-1
done
