#!/bin/bash
CLIENT_NUM=10       # SET SAME
WORKER_NUM=10       # SET SAME
SEED=42             # SET SAME
MODEL=LeNet5        # SET SAME
EPOCH=5             # SET SAME
BATCH_SIZE=512      # SET SAME
LR=0.01             # SET SAME    
DATASET=MNIST       # SET SAME
TRAIN_ITER=10       # SET SAME AS n_rounds for datasets?
CONCEPT_NUM=4       # TO DISCUSS
ROUND=1             # TO DISCUSS about train_iter
CL_ALGO=softclusterwin-1         # BASELINES     # change params for each type
CL_ALGO_ARG=hard-r    # BASELINES     # change params for each type
TIME_STRETCH=1
SERVER_NUM=1                 
CHANGE_POINTS=A
GPU_NUM_PER_SERVER=4
DATA_DIR="./../../../data/"
NOISE_PROB=0
CI=0
RESET_MODELS=0      
DRIFT_TOGETHER=0

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
           --seed $SEED \
           --change_points "${CHANGE_POINTS:-rand}"
done