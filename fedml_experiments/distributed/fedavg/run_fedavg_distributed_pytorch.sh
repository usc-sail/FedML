#!/usr/bin/env bash

CLIENT_NUM=118
WORKER_NUM=32
MODEL="onedcnnlstm"
DISTRIBUTION="onedcnnlstm"
ROUND=200
EPOCH=5
BATCH_SIZE=10
LR=0.01
DATASET="tiles"
DATA_DIR="./../../../data"
CLIENT_OPTIMIZER="sgd"
CI=0
output_dir="/data/rash/tiles-motif/expts/onedcnnlstm"

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3.7 ./main_fedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_config1_32" \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --ci $CI \
  --output_dir $output_dir
