#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

SAVE_PATH=models

#The first four parameters must be provided

MODE=$1
MODEL=$2
DATASET=$3
PROCESS=$4

FULL_DATA_PATH=$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"

echo $DATASET
python -u wrangle_KG.py $FULL_DATA_PATH

echo "Start Training......"

CUDA_VISIBLE_DEVICES=0 python -u main.py model $MODEL \
    input_drop 0.2 \
    hidden_drop 0.3 \
    feat_drop 0.2 \
    lr 0.003 \
    lr_decay 0.995 \
    dataset $DATASET \
    process $PROCESS \


if [ $MODE == "inject" ]
then

echo "Train The Inverter Network......"
CUDA_VISIBLE_DEVICES=0 python -u main_auto.py model $MODEL \
    input_drop 0.2 \
    hidden_drop 0.3 \
    feat_drop 0.2 \
    lr 0.003 \
    lr_decay 0.995 \
    dataset $DATASET \
    process $PROCESS \

echo "Identifying The Attacks......"
CUDA_VISIBLE_DEVICES=0 python -u main_inject.py model $MODEL \
    input_drop 0.2 \
    hidden_drop 0.3 \
    feat_drop 0.2 \
    lr 0.003 \
    lr_decay 0.995 \
    dataset $DATASET \
    process $PROCESS \

python -u wrangle_KG.py new_$DATASET
echo "Retrainin The Model......"
CUDA_VISIBLE_DEVICES=0 python -u main.py model $MODEL \
    input_drop 0.2 \
    hidden_drop 0.3 \
    feat_drop 0.2 \
    lr 0.003 \
    lr_decay 0.995 \
    dataset new_$DATASET \
    process $PROCESS \

elif [ $MODE == "remove" ]
then

echo "Identifying The Attacks......"
CUDA_VISIBLE_DEVICES=0 python -u main_remove.py model $MODEL \
    input_drop 0.2 \
    hidden_drop 0.3 \
    feat_drop 0.2 \
    lr 0.003 \
    lr_decay 0.995 \
    dataset $DATASET \
    process $PROCESS \

python -u wrangle_KG.py new_$DATASET
echo "Retrainin The Model......"
CUDA_VISIBLE_DEVICES=0 python -u main.py model $MODEL \
    input_drop 0.2 \
    hidden_drop 0.3 \
    feat_drop 0.2 \
    lr 0.003 \
    lr_decay 0.995 \
    dataset new_$DATASET \
    process $PROCESS \

else
   echo "Unknown MODE" $MODE
fi