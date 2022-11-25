# Import modules
import os
import gc
import h5py
import pickle
import logging
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
# Import PyTorch
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Import Huggingface
from transformers import BertTokenizerFast, BartTokenizerFast, T5TokenizerFast
# Import custom modules
from model.dataset import Seq2SeqDataset
from model.custom_transformer.transformer import Transformer
from model.custom_plm.T5 import custom_T5
from model.custom_plm.bart import custom_Bart
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name
from task.utils import model_save_name, results_save_name

def testing(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.tensorboard_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    write_log(logger, 'Start testing!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if args.tokenizer == 'spm':
        save_name = f'processed_{args.task}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.task}_{args.tokenizer}.hdf5'

    with h5py.File(os.path.join(save_path, 'test_' + save_name), 'r') as f:
        test_src_input_ids = f.get('test_src_input_ids')[:]
        test_src_attention_mask = f.get('test_src_attention_mask')[:]
        if args.task in ['translation', 'style_transfer', 'summarization']:
            test_trg_input_ids = f.get('test_trg_input_ids')[:]
            test_trg_attention_mask = f.get('test_trg_attention_mask')[:]
        elif args.task in ['multi-modal_classification']:
            test_src_img_path = f.get('test_src_img_path')[:]

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        src_vocab_num = len(src_word2id)
        if args.task in ['translation', 'style_transfer', 'summarization']:
            trg_word2id = data_['trg_word2id']
            trg_id2word = {v: k for k, v in trg_word2id.items()}
            trg_vocab_num = len(trg_word2id)
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Test setting============#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')

    variational_mode_dict = dict()
    if args.variational:
        variational_mode_dict['variational_model'] = args.variational_model
        variational_mode_dict['variational_token_processing'] = args.variational_token_processing
        variational_mode_dict['variational_with_target'] = args.variational_with_target
        variational_mode_dict['cnn_encoder'] = args.cnn_encoder
        variational_mode_dict['cnn_decoder'] = args.cnn_decoder
        variational_mode_dict['latent_add_encoder_out'] = args.latent_add_encoder_out
        variational_mode_dict['z_var'] = args.z_var
        variational_mode_dict['d_latent'] = args.d_latent

    if args.model_type == 'custom_transformer':
        model = Transformer(task=args.task,
                            src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                            pad_idx=args.pad_id, bos_idx=args.bos_id, eos_idx=args.eos_id,
                            d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                            dim_feedforward=args.dim_feedforward,
                            num_common_layer=args.num_common_layer, num_encoder_layer=args.num_encoder_layer,
                            num_decoder_layer=args.num_decoder_layer,
                            src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                            dropout=args.dropout, embedding_dropout=args.embedding_dropout,
                            trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                            emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing, 
                            variational=args.variational,
                            variational_mode_dict=variational_mode_dict,
                            parallel=args.parallel)
        tgt_subsqeunt_mask = model.generate_square_subsequent_mask(args.trg_max_len - 1, device)
    # elif args.model_type == 'T5':
    #     model = custom_T5(isPreTrain=args.isPreTrain, d_latent=args.d_latent, 
    #                       variational_mode=args.variational_mode, z_var=args.z_var,
    #                       decoder_full_model=True)
    #     tgt_subsqeunt_mask = None
    elif args.model_type == 'bart':
        model = custom_Bart(task=args.task,
                            isPreTrain=args.isPreTrain, variational=args.variational,
                            variational_mode_dict=variational_mode_dict,
                            src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                            emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing)
        tgt_subsqeunt_mask = None
    elif args.model_type == 'bert':
        model = custom_Bert(task=args.task, num_class=128, # Need to refactoring
                            isPreTrain=args.isPreTrain, variational=args.variational,
                            src_language=src_language,
                            variational_mode_dict=variational_mode_dict,
                            src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                            emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing)
        tgt_subsqeunt_mask = None
    model = model.to(device)

    # lode model
    save_file_name = model_save_name(args)
    model.load_state_dict(torch.load(save_file_name)['model'])
    model = model.eval()
    write_log(logger, f'Loaded model from {save_file_name}')

    # 2) Dataloader setting
    test_dataset = Seq2SeqDataset(src_list=test_src_input_ids, src_att_list=test_src_attention_mask,
                                  trg_list=test_trg_input_ids, trg_att_list=test_trg_attention_mask,
                                  src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                                  pad_idx=model.pad_idx, eos_idx=model.eos_idx)
    test_dataloader = DataLoader(test_dataset, drop_last=False, batch_size=args.test_batch_size, shuffle=False,
                                 pin_memory=True, num_workers=args.num_workers)
    write_log(logger, f"Total number of trainingsets  iterations - {len(test_dataset)}, {len(test_dataloader)}")

    # 3) Load tokenizer
    if args.tokenizer == 'bart':
        tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    elif args.tokenizer == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    elif args.tokenizer == 'T5':
        tokenizer = T5TokenizerFast.from_pretrained('t5-base')
    else:
        preprocess_save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
        spm_model = spm.SentencePieceProcessor()
        spm_model.Load(f'{preprocess_save_path}/m_src_{args.sentencepiece_model}_{args.trg_vocab_size}.model')

    # 4) Define array to save results
    source_sentences = []
    predicted_sentences = []
    target_sentences = []
    #source_tokens = []
    predicted_tokens = []
    target_tokens = []

    #===================================#
    #============Inference==============#
    #===================================#

    with torch.no_grad():
        for i, batch_iter in enumerate(tqdm(test_dataloader, bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            # Input, output setting
            src_sequence = batch_iter[0]
            src_att = batch_iter[1]
            trg_sequence = batch_iter[2]
            trg_att = batch_iter[3]

            src_sequence = src_sequence.to(device, non_blocking=True)
            src_att = src_att.to(device, non_blocking=True)
            trg_sequence = trg_sequence.to(device, non_blocking=True)
            trg_att = trg_att.to(device, non_blocking=True)

            predicted = model.generate(src_sequence, src_att, 
                                       beam_size=args.beam_size, beam_alpha=args.beam_alpha, 
                                       repetition_penalty=args.repetition_penalty, device=device)

            for j, predicted_sequence in enumerate(predicted):
                src_seq_list = src_sequence.cpu().tolist()
                trg_seq_list = trg_sequence.cpu().tolist()

                if args.tokenizer == 'spm':
                    source = spm_model.DecodeIds(src_seq_list[j])
                    predicted = spm_model.DecodeIds(predicted_sequence)
                    target = spm_model.DecodeIds(trg_seq_list[j])
                else:
                    source = tokenizer.decode(src_seq_list[j], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    predicted = tokenizer.decode(predicted_sequence, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                    target = tokenizer.decode(trg_seq_list[j], skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if args.use_tensorboard:
                    writer.add_text('TEST/Source', source, (i+1)*(j+1))
                    writer.add_text('TEST/Predicted', predicted, (i+1)*(j+1))
                    writer.add_text('TEST/Target', target, (i+1)*(j+1))

                # save sentences
                source_sentences.append(source)
                predicted_sentences.append(predicted)
                target_sentences.append(target)
                
                # Get BLEU score
                predicted_token = [trg_id2word[idx] for idx in predicted_sequence]
                target_token = [trg_id2word[idx] for idx in trg_sequence.cpu().tolist()[j]]
                target_tokens.append([target_token])
                predicted_tokens.append(predicted_token)
                corpus_bleu_score = corpus_bleu(target_tokens, predicted_tokens)

                if args.use_tensorboard:
                    writer.add_scalar('TEST/Corpus BLEU', corpus_bleu_score, i)
            
    final_bleu_score = corpus_bleu(target_tokens, predicted_tokens)
    write_log(logger, f'[TEST] Final BLEU score: {final_bleu_score}')
    if args.use_tensorboard:
        writer.add_text('TEST/Final BLEU', str(final_bleu_score))

    #===================================#
    #=============Saving================#
    #===================================#
    
    # Save sentences to csv file
    result_path = results_save_name(args)
    
    # Make pandas dataframe with source_sentences, predicted_sentences, target_sentences
    df = pd.DataFrame(
        {'source': source_sentences, 
        'predicted': predicted_sentences, 
        'target': target_sentences}
    )
    df.to_csv(result_path, index=False)