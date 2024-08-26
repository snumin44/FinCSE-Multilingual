#!/bin/sh

# If you wanna use more gpus, set GPU_ID like "0, 1, 2"

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 sentence_retrieval.py \
    	--model '../output/bert_checkpoint' \
        --tokenizer '../output/bert_checkpoint' \
        --faiss_path '../pickles/faiss.index' \
        --sent_path '../pickles/sentence.pkl' \
        --search_k 10 \
        --return_k 10 \
        --pooler 'cls'