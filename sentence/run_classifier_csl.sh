#!/usr/bin/env bash

TASK_NAME="csl"
MODEL_NAME="bert-base-chinese"
# CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

CURRENT_DIR="workspace"

export CUDA_VISIBLE_DEVICES="0"

export GLUE_DATA_DIR=$CURRENT_DIR/CLUEdatasets
export MODEL_NAME=$CURRENT_DIR/checkpoint/bert

## training cross distill
#export MODEL_NAME=$CURRENT_DIR/checkpoint/albert_tiny
export TEACHER_MODEL_NAME=$CURRENT_DIR/csl_output/bert/checkpoint-5000


# make output dir
if [ ! -d $CURRENT_DIR/${TASK_NAME}_output ]; then
  mkdir -p $CURRENT_DIR/${TASK_NAME}_output
  echo "makedir $CURRENT_DIR/${TASK_NAME}_output"
fi

# run task
# cd $CURRENT_DIR
echo "Start running..."
 if [ $# == 0 ]; then
     python run_teacher_classifier.py \
       --model_type=albert \
       --model_name_or_path=$MODEL_NAME \
       --task_name=$TASK_NAME \
       --do_train \
       --do_eval \
       --do_lower_case \
       --data_dir=$GLUE_DATA_DIR/${TASK_NAME}/ \
       --max_seq_length=256 \
       --per_gpu_train_batch_size=8 \
       --per_gpu_eval_batch_size=8 \
       --learning_rate=3e-5 \
       --num_train_epochs=30.0 \
       --logging_steps=250 \
       --save_steps=250 \
       --output_dir=$CURRENT_DIR/${TASK_NAME}_output/ \
       --overwrite_output_dir \
       --seed=42
 fi



if [ $# == 0 ]; then
    python run_classifier_cross_distill.py \
        --model_type=albert \
        --model_name_or_path=$MODEL_NAME \
        --teacher_model_name_or_path=$TEACHER_MODEL_NAME \
        --task_name=$TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir=$GLUE_DATA_DIR/${TASK_NAME}/ \
        --max_seq_length=256 \
        --per_gpu_train_batch_size=8 \
        --per_gpu_eval_batch_size=8 \
        --learning_rate=1e-4 \
        --teacher_learning_rate=1e-6 \
        --num_train_epochs=30.0 \
        --logging_steps=250 \
        --save_steps=250 \
        --output_dir=$CURRENT_DIR/${TASK_NAME}_output/ \
        --overwrite_output_dir \
        --is_load_classifier='True' \
        --seed=42 \
        --is_KD_mse_loss \
        --teacher_hard_loss_alpha=20.0 \
        --teacher_cls_loss_alpha=5.0 \
        --teacher_logtic_loss_alpha=1.0 \
        --stu_hard_loss_alpha=1.0 \
        --stu_cls_loss_alpha=0.0 \
        --stu_logtic_loss_alpha=1.0
fi
