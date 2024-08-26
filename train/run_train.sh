#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--model 'google-bert/bert-base-multilingual-cased' \
        --tokenizer 'google-bert/bert-base-multilingual-cased' \
    	--train_data '../data/finance_multilingual_train.csv' \
        --valid_data '../data/finance_multilingual_valid.csv' \
    	--output_path '../output/checkpoint' \
    	--epochs 1 \
        --batch_size 256 \
        --max_length 64 \
        --dropout 0.1 \
        --pooler 'cls' \
        --eval_strategy 'steps' \
        --eval_step 100 \
        --amp