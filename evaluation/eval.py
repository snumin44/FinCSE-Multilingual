import os
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd

import torch

import sys
sys.path.append("../") 

from src.data_loader import Dataset_CSV, Dataset_STS, DataCollator
from src.model import SimCSE
from evaluation.utils import inference 

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='evaluate simcse')

    # Required
    parser.add_argument('--model', type=str, required=True,
                        help='Directory of pretrained model'
                       )    
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Directory of pretrained tokenizer'
                       )     
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path of validation set.'
                       )
    parser.add_argument('--save_pickle', action='store_false', default=True,
                        help='Save faiss index and context pickle'
                       )

    # Tokenizer & Collator settings
    parser.add_argument('--max_length', default=64, type=int,
                        help='Max length of sequence'
                       )
    parser.add_argument('--padding', action="store_false",default=True,
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_false", default=True,
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--shuffle', action="store_false", default=True,
                        help='Load shuffled sequences'
                       )

    # Train config       
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )        
    parser.add_argument('--temp', default=0.05, type=float,
                        help='Temperature for similarity'
                       )    
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout ratio'
                       )  
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default = 42, type=int,
                        help = 'Random seed'
                       )  
    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    init_logging()
    seed_everything(args)
    
    LOGGER.info('*** Evaluate Multilingual SimCSE ***')
    LOGGER.info(f'Evaluating Performance with \'{args.test_data}\'')
    encoder = SimCSE(args).to(args.device)
    test_dataset = Dataset_CSV.load_dataset(args.test_data)
    scores = inference(encoder, test_dataset, args)
    
    print()
    print('=== Top-k Accuracy ===')
    print(f"Top1 Acc: {scores['top1_accuracy']*100:.2f} (%)")
    print(f"Top5 Acc: {scores['top5_accuracy']*100:.2f} (%)")
    print(f"Top10 Acc: {scores['top10_accuracy']*100:.2f} (%)")
    print('======================')    

if __name__ == '__main__':
    args = argument_parser()
    main(args)