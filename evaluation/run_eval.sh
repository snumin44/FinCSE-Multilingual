#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 eval.py \
    	--model '../output/bert_checkpoint' \
        --tokenizer '../output/bert_checkpoint' \
        --test_data '../data/finance_multilingual_valid.csv' \
        --batch_size 256 \
        --max_length 64 \
        --pooler 'cls'