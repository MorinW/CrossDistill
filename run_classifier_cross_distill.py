# -*- coding: utf-8 -*-
# @Author: chujun.chu
# @Date:   2023-12.30
""" Finetuning the library models for sequence classification on CLUE (Bert, ERNIE, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, BertConfig, AlbertConfig,
                          BertTokenizer, BertTokenizerFast,
                          XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer
                          )
from pretrain_models_pooler import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from metrics.clue_compute_metrics import compute_metrics
from processors import clue_output_modes as output_modes
from processors import clue_processors as processors
from processors import clue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn, xlnet_collate_fn
from tools_.common import seed_everything, save_numpy
from tools_.common import init_logger, logger
from tools_.progressbar import ProgressBar
from collections import OrderedDict
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig,
#                                                                                 RobertaConfig)), ())
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, BertTokenizerFast)
}

def loss_fn_kd(student_outputs, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    ce_distill_loss = F.kl_div(
        input=F.log_softmax(
            student_outputs / (2./3), dim=-1), #! logits: [32,3]
        target=F.softmax(
            teacher_outputs / (2./3), dim=-1), #! distill_temp: 2.0
        reduction="batchmean") * ((2./3) ** 2)

    return ce_distill_loss

def train(args, train_dataset, student_model, teacher_model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    stu_optimizer_grouped_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    stu_optimizer = AdamW(stu_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    stu_scheduler = get_linear_schedule_with_warmup(stu_optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    tea_optimizer_grouped_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        # {"params": [p for n, p in teacher_model.named_parameters() if any(nd in n for nd in no_trm_lr) and not any(nd in n for nd in no_decay)], "lr": 1e-6, 'weight_decay': 0.01}
    ]
    tea_optimizer = AdamW(tea_optimizer_grouped_parameters, lr=args.teacher_learning_rate, eps=args.adam_epsilon)
    tea_scheduler = get_linear_schedule_with_warmup(tea_optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

    # # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    loss_mse = MSELoss()
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    student_model.zero_grad()
    teacher_model.zero_grad()
    student_model.to(args.device)
    teacher_model.to(args.device)
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for ep in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            ada_prob = 1.0
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'albert',
                                                                           'roberta'] else None  # XLM, DistilBERT don't use segment_ids\]
            # # first stage train teacher model
            if args.is_cross_distill:
                print('training teacher model***************************')
                with torch.no_grad():
                    frozen_outputs = student_model(**inputs, ada_prob=ada_prob)
                outputs = teacher_model(**inputs, ada_prob=ada_prob)
                student_model.eval()
                teacher_model.train()

                ## loss
                hard_loss = outputs.loss
                cls_loss = loss_mse(outputs.first_token_tensor, frozen_outputs.first_token_tensor)
                # emb_loss = loss_mse(outputs[2], frozen_outputs[2])
                if args.is_KD_mse_loss:
                    logtic_loss = loss_mse(outputs.logits, frozen_outputs.logits)
                else:
                    logtic_loss = loss_fn_kd(outputs.logits, frozen_outputs.logits)
                
                loss = hard_loss * args.teacher_hard_loss_alpha + cls_loss * args.teacher_cls_loss_alpha + logtic_loss * args.teacher_logtic_loss_alpha

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), args.max_grad_norm)
                pbar(step, {'tea_loss': loss.item(), 'hard_loss': hard_loss.item(), 'cls_loss': cls_loss.item()})
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    tea_optimizer.step()
                    tea_scheduler.step()  # Update learning rate schedule
                    teacher_model.zero_grad()

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        print(" ")
                        # Log metrics
                        if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                            evaluate(args, teacher_model, tokenizer)


            # # second stage train student model
            with torch.no_grad():
                frozen_outputs = teacher_model(**inputs, ada_prob=ada_prob)
            outputs = student_model(**inputs, ada_prob=ada_prob)
            teacher_model.eval()
            student_model.train()

            ## loss
            hard_loss = outputs.loss
            cls_loss = loss_mse(outputs.first_token_tensor, frozen_outputs.first_token_tensor)
            # emb_loss = loss_mse(outputs[2], frozen_outputs[2])
            if args.is_KD_mse_loss:
                logtic_loss = loss_mse(outputs.logits, frozen_outputs.logits)
            else:
                logtic_loss = loss_fn_kd(outputs.logits, frozen_outputs.logits)
            # loss = hard_loss + repo_loss * 2
            loss = hard_loss*args.stu_hard_loss_alpha + cls_loss*args.stu_cls_loss_alpha + logtic_loss*args.stu_logtic_loss_alpha

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            pbar(step, {'stu_loss': loss.item(), 'hard_loss': hard_loss.item(), 'cls_loss': cls_loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                stu_optimizer.step()
                stu_scheduler.step()  # Update learning rate schedule
                student_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(" ")
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, student_model, tokenizer)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = student_model.module if hasattr(student_model,
                                                                    'module') else student_model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=xlnet_collate_fn if args.model_type in ['xlnet'] else collate_fn)

        # Eval!
        logger.info("********* Running evaluation {} ********".format(prefix))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet', 'albert',
                                                                               'roberta'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss = outputs[0]
                logits = outputs[1]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("******** Eval results {} ********".format(prefix))
        for key in sorted(result.keys()):
            logger.info(" dev: %s = %s", key, str(result[key]))
    return results


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        'model',
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and 'roberta' in args.model_type:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='albert', type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path",
                        type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ''))
    parser.add_argument("--teacher_model_name_or_path",
                        type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ''))
    parser.add_argument("--task_name", default='cmnli', type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--teacher_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--train_student', default='stu|tea',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--is_load_classifier', type=str, default='True', 
                        help="is_load_classifier.")
    parser.add_argument('--is_KD_mse_loss', action='store_true', 
                        help="is_KD_mse_loss.")
    parser.add_argument('--is_cross_distill', action='store_true', 
                        help="is_cross_distill.")

    parser.add_argument('--teacher_hard_loss_alpha', type=float, default=1.0, help="teacher_hard_loss_alpha.")
    parser.add_argument('--teacher_cls_loss_alpha', type=float, default=1.0, help="teacher_cls_loss_alpha.")
    parser.add_argument('--teacher_logtic_loss_alpha', type=float, default=1.0, help="teacher_logtic_loss_alpha.")
    
    parser.add_argument('--stu_hard_loss_alpha', type=float, default=1.0, help="stu_hard_loss_alpha.")
    parser.add_argument('--stu_cls_loss_alpha', type=float, default=1.0, help="stu_cls_loss_alpha.")
    parser.add_argument('--stu_logtic_loss_alpha', type=float, default=1.0, help="stu_logtic_loss_alpha.")


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    init_logger(log_file=args.output_dir + '/{}-{}.log'.format(args.model_type, args.task_name))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    seed_everything(args.seed)
    # Prepare CLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    print(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    # load teacher model
    teacher_config = BertConfig.from_pretrained(args.teacher_model_name_or_path,
                                                num_labels=num_labels, finetuning_task=args.task_name)
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model_name_or_path,
                                                                  from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                  config=teacher_config)

    teacher_state_dict = teacher_model.state_dict()
    new_state_dict = OrderedDict()

    if args.is_load_classifier == 'True':
        logger.info("is_load_classifier %s", 'True')

        # load pooler & classifier
        for key, value in teacher_state_dict.items():
            if 'pooler.dense' in key:
                new_state_dict[key.replace('bert.pooler.dense', 'albert.pooler')] = value

        for key, value in teacher_state_dict.items():
            if 'pooler_adapter' in key:
                new_state_dict[key.replace('bert', 'albert')] = value

        for key, value in teacher_state_dict.items():
            if 'classifier' in key:
                new_state_dict[key] = value

        # load other parm
        for key, value in model.state_dict().items():
            if 'classifier' not in key and 'pooler' not in key:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict)
        for k, v in model.named_parameters():
            if 'classifier' in k or 'pooler' in k:
                v.requires_grad = False
                # logger.info("Training/stu parameters %s%s", k, v)

        ## teacher_classifier_frozen
        for k, v in teacher_model.named_parameters():
            if 'classifier' in k or 'pooler' in k:
                v.requires_grad = False
                # logger.info("Training/tea parameters %s%s", k, v)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, teacher_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("%s = %s\n" % (key, str(results[key])))
    logger.info('################################################################')


if __name__ == "__main__":
    main()
