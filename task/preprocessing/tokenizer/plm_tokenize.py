import argparse
from transformers import  BertTokenizer, BartTokenizer, T5Tokenizer

from datasets import load_dataset

def plm_tokenizing(sequence_dict: dict, args: argparse.Namespace, 
                   domain: str = 'src'):

    # 1) Pre-setting
    processed_sequences = dict()
    processed_sequences['train'] = dict()
    processed_sequences['valid'] = dict()
    processed_sequences['test'] = dict()

    if domain == 'src':
        max_len = args.src_max_len
    if domain == 'trg':
        max_len = args.trg_max_len

    if args.tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif args.tokenizer == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif args.tokenizer == 'T5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    else:
        raise Exception(f'{language} language does not support')

    for phase in ['train', 'valid', 'test']:
        encoded_dict = \
        tokenizer(
            sequence_dict[phase],
            max_length=max_len,
            padding='max_length',
            truncation=True
        )
        processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
        processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']
        if args.tokenizer == 'bert':
            processed_sequences[phase]['token_type_ids'] = encoded_dict['token_type_ids']

    # BART's decoder input id need to start with 'model.config.decoder_start_token_id'
    if args.tokenizer == 'bart' and domain == 'trg':
        for i in range(len(processed_sequences[phase]['input_ids'])):
            processed_sequences[phase]['input_ids'][i][0] = 2
    
    word2id = tokenizer.get_vocab()

    return processed_sequences, word2id