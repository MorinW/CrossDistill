#!/usr/bin/env bash
TASK_NAME="CHID"

CURRENT_DIR="workspace"
export MODEL_NAME=bert
export OUTPUT_DIR=$CURRENT_DIR
export GLUE_DIR=$CURRENT_DIR/CLUEdatasets/ # set your data dir

export BERT_DIR=$CURRENT_DIR/checkpoint/bert
export TEACHER_DIR=$CURRENT_DIR/CMRC2018_output_bert_base
export STUDENT_DIT=$CURRENT_DIR/checkpoint/albert_tiny

# make output dir
if [ ! -d $CURRENT_DIR/${TASK_NAME}_output ]; then
  mkdir -p $CURRENT_DIR/${TASK_NAME}_output
  echo "makedir $CURRENT_DIR/${TASK_NAME}_output"
fi

python run_teacher_mrc.py \
   --gpu_ids="0" \
   --train_epochs=6 \
   --n_batch=32 \
   --lr=3e-5 \
   --warmup_rate=0.1 \
   --max_seq_length=512 \
   --task_name=$TASK_NAME \
   --vocab_file=$BERT_DIR/vocab.txt \
   --bert_config_file=$BERT_DIR/config.json \
   --init_restore_dir=$BERT_DIR/pytorch_model.bin \
   --train_dir=$GLUE_DIR/$TASK_NAME/train_features.json \
   --train_file=$GLUE_DIR/$TASK_NAME/train.json \
   --dev_dir1=$GLUE_DIR/$TASK_NAME/dev_examples.json \
   --dev_dir2=$GLUE_DIR/$TASK_NAME/dev_features.json \
   --dev_file=$GLUE_DIR/$TASK_NAME/dev.json \
   --checkpoint_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
   --output_dir=$CURRENT_DIR/${TASK_NAME}_output



python run_mrc_cross_distill.py \
  --gpu_ids="0" \
  --train_epochs=6 \
  --n_batch=32 \
  --lr=3e-4 \
  --teacher_lr=1e-7 \
  --warmup_rate=0.1 \
  --max_seq_length=512 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/config.json \
  --init_restore_dir=$BERT_DIR/pytorch_model.bin \
  --train_dir=$GLUE_DIR/$TASK_NAME/train_features.json \
  --train_file=$GLUE_DIR/$TASK_NAME/train.json \
  --dev_dir1=$GLUE_DIR/$TASK_NAME/dev_examples.json \
  --dev_dir2=$GLUE_DIR/$TASK_NAME/dev_features.json \
  --dev_file=$GLUE_DIR/$TASK_NAME/dev.json \
  --checkpoint_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --teacher_model_name_or_path=$TEACHER_DIR \
  --student_model_name_or_path=$STUDENT_DIT



