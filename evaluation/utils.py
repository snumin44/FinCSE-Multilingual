import os
import faiss
import torch
import pickle
import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaTokenizerFast, XLMRobertaConfig

def inference(encoder, test_dataset, args):
    
    if 'xlm-roberta' in args.model:
        config = XLMRobertaConfig.from_pretrained(args.model)
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.tokenizer)  
    else:
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, config=config)
    
    sent0 = test_dataset.sent0
    sent1 = test_dataset.sent1
    
    answer = [idx for idx in range(len(sent0))]
    
    assert len(sent0) == len(sent1)
    assert len(sent0) == len(answer)
    
    sents = sent0 + sent1

    sent_embeddings = []   
    
    encoder.eval() 
    for start_index in range(0, len(sents), args.batch_size):
        # Divide Sentences into Mini-Batch
        batch = sents[start_index : start_index + args.batch_size]
           
        features = tokenizer(batch,
                             padding=args.padding,
                             max_length=args.max_length,
                             truncation=args.truncation,)
        
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction.
        with torch.no_grad():
            pooler_output = encoder.get_embeddings(input_ids=torch.tensor(features['input_ids']).to(args.device),
                                                   attention_mask=torch.tensor(features['attention_mask']).to(args.device),)
                                                   #token_type_ids=torch.tensor(features['token_type_ids']).to(args.device),)
        sent_embeddings.extend(pooler_output.cpu())
    
    sent_embeddings = np.asarray([emb.numpy() for emb in sent_embeddings]) 
    sent0_embeddings, sent1_embeddings, = sent_embeddings[:len(sent0)], sent_embeddings[len(sent0):]
    assert len(sent0_embeddings) == len(sent1_embeddings)

    faiss.normalize_L2(sent1_embeddings)       
    index = faiss.IndexFlatIP(config.hidden_size)
    index.add(sent1_embeddings)    

    if args.save_pickle:
        save_path = '../pickles'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        faiss_path = os.path.join(save_path, 'faiss.index')
        faiss.write_index(index, faiss_path)
        
        sent_path = os.path.join(save_path, 'sentence.pkl')
        with open(sent_path, 'wb') as file:
                pickle.dump({
                    'sentence': sent1
                }, file)
            
    D, I = index.search(sent0_embeddings, k=100)

    scores = get_topk_accuracy(I, answer)

    return scores


def get_topk_accuracy(faiss_index, answer_index): 

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    top20_correct = 0
    top50_correct = 0
    top100_correct = 0
    
    for idx, answer in enumerate(answer_index):
        
        #  *** faiss index: (question num * k) ***
        #      [[73587  2746 15265 96434 ...]
        #       [98388 13550 93912 92610 ...]
        #                    ...
        #       [97530 93498 16607 98168 ...]
        #       [52308 24908 70869 20824 ...]
        #       [44597 35140  7572  4596 ...]]
         
        retrieved_idx = faiss_index[idx] 
         
        if any(ridx == answer for ridx in retrieved_idx[:1]):
            top1_correct += 1
        if any(ridx == answer for ridx in retrieved_idx[:5]):
            top5_correct += 1
        if any(ridx == answer for ridx in retrieved_idx[:10]):
            top10_correct += 1
        if any(ridx == answer for ridx in retrieved_idx[:20]):
            top20_correct += 1
        if any(ridx == answer for ridx in retrieved_idx[:50]):
            top50_correct += 1
        if any(ridx == answer for ridx in retrieved_idx[:100]):
            top100_correct += 1
        
    top1_accuracy = top1_correct / len(answer_index)
    top5_accuracy = top5_correct / len(answer_index)
    top10_accuracy = top10_correct / len(answer_index)    
    top20_accuracy = top20_correct / len(answer_index)
    top50_accuracy = top50_correct / len(answer_index)
    top100_accuracy = top100_correct / len(answer_index)

    return {
        'top1_accuracy':top1_accuracy,
        'top5_accuracy':top5_accuracy,
        'top10_accuracy':top10_accuracy,        
        'top20_accuracy':top20_accuracy,
        'top50_accuracy':top50_accuracy,
        'top100_accuracy':top100_accuracy,
    }