import os
import torch
from torch.nn import functional as F

def input_to_device(args, batch_iter, device):

    src_sequence = batch_iter[0]
    src_att = batch_iter[1]
    trg_sequence = batch_iter[2]
    trg_att = batch_iter[3]

    src_sequence = src_sequence.to(device, non_blocking=True)
    src_att = src_att.to(device, non_blocking=True)
    trg_sequence = trg_sequence.to(device, non_blocking=True)
    trg_att = trg_att.to(device, non_blocking=True)

    return src_sequence, src_att, trg_sequence, trg_att

def model_save_name(args):
    save_path = os.path.join(args.model_save_path, args.data_name, args.tokenizer, args.model_type)
    save_name_pre = 'checkpoint'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # SentencePiece
    if args.tokenizer == 'spm':
        save_name_pre += f'_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}'

    # Variational
    if args.variational:
        save_name_pre += f'_v_{args.variational_model}'
        save_name_pre += f'_token_{args.variational_token_processing}'
        save_name_pre += f'_with_target_{args.variational_with_target}'
        save_name_pre += f'_cnn_encoder_{args.cnn_encoder}'
        save_name_pre += f'_cnn_decoder_{args.cnn_decoder}'
        save_name_pre += f'_latent_add_{args.latent_add_encoder_out}'
        
    save_name_pre += '.pth.tar'
    save_file_name = os.path.join(save_path, save_name_pre)
    return save_file_name

def results_save_name(args):
    if not os.path.exists(os.path.join(args.result_path)):
        os.mkdir(os.path.join(args.result_path))
    if not os.path.exists(os.path.join(args.result_path, args.data_name)):
        os.mkdir(os.path.join(args.result_path, args.data_name))
    result_path_ = os.path.join(args.result_path, args.data_name, args.tokenizer)
    if not os.path.exists(result_path_):
        os.mkdir(result_path_)
    if args.tokenizer == 'spm':
        save_name_pre = f'Result_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational_mode}_p_{args.parallel}.csv'
    else:
        save_name_pre = f'Result_v_{args.variational_mode}_p_{args.parallel}.csv'
    save_result_name = os.path.join(result_path_, save_name_pre)

    return save_result_name