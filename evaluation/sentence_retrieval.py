import faiss
import sys
import torch
import pickle
import argparse
import numpy as np
from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        XLMRobertaTokenizerFast,
        XLMRobertaConfig,
)

sys.path.append('../')
from src.model import Pooler

def argument_parser():

    parser = argparse.ArgumentParser(description='get topk-accuracy of retrieval model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path of multilingual simcse model'
                       )
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path of multilingual simcse tokenizer'
                       )
    parser.add_argument('--faiss_path', type=str, required=True,
                        help='Path of faiss pickle'
                       )
    parser.add_argument('--sent_path', type=str, required=True,
                        help='Path of faiss pickle'
                       )
    parser.add_argument('--search_k', default=50, type=int,
                        help='Number of retrieved documents'
                       )
    parser.add_argument('--return_k', default=10, type=int,
                        help='Number of returned documents'
                       )   
    parser.add_argument('--max_length', default=512, type=int,
                        help='Max length of sequence'
                       )                        
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )
    parser.add_argument('--truncation', action="store_false", default=True,
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )    
    
    args = parser.parse_args()
    return args


def inference(question, q_encoder, tokenizer, faiss_index, sentence,
              search_k=2000, return_k=5, max_length=512, pooler=None, truncation=True, device='cuda'):
    
    q_encoder = q_encoder.to(device)
               
    features = tokenizer(question, max_length=max_length, truncation=truncation, return_tensors='pt').to(device)

    q_encoder.eval()
    with torch.no_grad():
        q_output = q_encoder(**features, return_dict=True)

    pooler_output = pooler(features['attention_mask'], q_output)
    pooler_output = pooler_output.cpu().detach().numpy() # (1, 768)

    D, I = faiss_index.search(pooler_output, search_k)

    return D, I


def main(args):
    
    # Load model & tokenizer
    q_encoder = AutoModel.from_pretrained(args.model)
    
    if 'xlm-roberta' in args.model:
        config = XLMRobertaConfig.from_pretrained(args.model)
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.tokenizer)
    else:
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, config=config)
    
    pooler = Pooler(args.pooler)
    
    # Load faiss index and context
    faiss_index = faiss.read_index(args.faiss_path)

    with open(args.sent_path, 'rb') as file:
        sentence_pickle = pickle.load(file)
        sentence = sentence_pickle.get('sentence', None)

    # Retrieval loop.
    while True:
        input_text = input('Please enter the sentence you want to search for: (type "exit" to quit): ')
        
        if input_text.lower() == "exit":
            print("Exiting the inference loop.")
            break
        
        D, I = inference(input_text, q_encoder, tokenizer, faiss_index, sentence,
                         search_k=args.search_k, return_k=args.return_k, max_length=args.max_length,
                         pooler=pooler, truncation=args.truncation, device=args.device
                        )
            
        for idx, (distance, index) in enumerate(zip(D[0], I[0])):
            print()
            print(f'|| Retrieval Ranking: {idx+1} || Similarity Score: {distance:.2f} ||')
            print('================================================================================')
            print(sentence[index].replace('\n', ' '))
            print()

            if idx + 1 == args.return_k:
                break

if __name__ == '__main__':
    args = argument_parser()
    main(args)