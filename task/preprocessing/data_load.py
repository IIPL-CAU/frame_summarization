import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset

def data_split_index(seq, valid_ration: float = 0.1, test_ratio: float = 0.03):

    paired_data_len = len(seq)
    valid_num = int(paired_data_len * valid_ration)
    test_num = int(paired_data_len * test_ratio)

    valid_index = np.random.choice(paired_data_len, valid_num, replace=False)
    train_index = list(set(range(paired_data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, test_num, replace=False)
    train_index = list(set(train_index) - set(test_index))

    return train_index, valid_index, test_index

def total_data_load(args):

    src_list = dict()
    trg_list = dict()

    #===================================#
    #============Translation============#
    #===================================#

    # WMT2016 Multimodal [DE -> EN]

    if args.data_name == 'WMT2016_Multimodal':
        args.data_path = os.path.join(args.data_path,'WMT/2016/multi_modal')

        # 1) Train data load
        with open(os.path.join(args.data_path, 'train.de'), 'r') as f:
            src_list['train'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'train.en'), 'r') as f:
            trg_list['train'] = [x.replace('\n', '') for x in f.readlines()]

        # 2) Valid data load
        with open(os.path.join(args.data_path, 'val.de'), 'r') as f:
            src_list['valid'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'val.en'), 'r') as f:
            trg_list['valid'] = [x.replace('\n', '') for x in f.readlines()]

        # 3) Test data load
        with open(os.path.join(args.data_path, 'test.de'), 'r') as f:
            src_list['test'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'test.en'), 'r') as f:
            trg_list['test'] = [x.replace('\n', '') for x in f.readlines()]

    # WMT2014 Translation [DE -> EN]
        
    elif args.data_name == 'WMT2014_de_en':
        args.data_path = os.path.join(args.data_path,'WMT/2014/de_en')

        # 1) Train data load
        with open(os.path.join(args.data_path, 'train.de'), 'r') as f:
            src_list['train'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'train.en'), 'r') as f:
            trg_list['train'] = [x.replace('\n', '') for x in f.readlines()]

        # 2) Valid data load
        with open(os.path.join(args.data_path, 'val.de'), 'r') as f:
            src_list['valid'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'val.en'), 'r') as f:
            trg_list['valid'] = [x.replace('\n', '') for x in f.readlines()]

        # 3) Test data load
        with open(os.path.join(args.data_path, 'test.de'), 'r') as f:
            src_list['test'] = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'test.en'), 'r') as f:
            trg_list['test'] = [x.replace('\n', '') for x in f.readlines()]

    # elif args.data_name == 'shift_challenge':
    #     args.data_path = os.path.join(args.data_path,'shift_challenge')

    # Korpora [EN -> KR]

    elif args.data_name == 'korpora':
        args.data_path = os.path.join(args.data_path,'korpora')

        en = pd.read_csv(os.path.join(args.data_path, 'pair_eng.csv'), names=['en'])['en']
        kr = pd.read_csv(os.path.join(args.data_path, 'pair_kor.csv'), names=['kr'])['kr']

        train_index, valid_index, test_index = data_split_index(en)

        src_list['train'] = [en[i] for i in train_index]
        trg_list['train'] = [kr[i] for i in train_index]

        src_list['valid'] = [en[i] for i in valid_index]
        trg_list['valid'] = [kr[i] for i in valid_index]

        src_list['test'] = [en[i] for i in test_index]
        trg_list['test'] = [kr[i] for i in test_index]

    # AIHUB [EN -> KR]

    elif args.data_name == 'aihub_en_kr':
        args.data_path = os.path.join(args.data_path,'AI_Hub_KR_EN')

        dat = pd.read_csv(os.path.join(args.data_path, '1_구어체(1).csv'))

        train_index, valid_index, test_index = data_split_index(dat)

        src_list['train'] = [dat['EN'][i] for i in train_index]
        trg_list['train'] = [dat['KR'][i] for i in train_index]

        src_list['valid'] = [dat['EN'][i] for i in valid_index]
        trg_list['valid'] = [dat['KR'][i] for i in valid_index]

        src_list['test'] = [dat['EN'][i] for i in test_index]
        trg_list['test'] = [dat['KR'][i] for i in test_index]

    #===================================#
    #========Text Style Transfer========#
    #===================================#

    # GYAFC [Informal -> Formal]

    if args.data_name == 'GYAFC':
        args.data_path = os.path.join(args.data_path,'GYAFC_Corpus')

        # 1) Train data load
        with open(os.path.join(args.data_path, 'Entertainment_Music/train/informal_em_train.txt'), 'r') as f:
            music_src = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'Entertainment_Music/train/formal_em_train.txt'), 'r') as f:
            music_trg = [x.replace('\n', '') for x in f.readlines()]

        with open(os.path.join(args.data_path, 'Family_Relationships/train/informal_fr_train.txt'), 'r') as f:
            family_src = [x.replace('\n', '') for x in f.readlines()]
        with open(os.path.join(args.data_path, 'Family_Relationships/train/formal_fr_train.txt'), 'r') as f:
            family_trg = [x.replace('\n', '') for x in f.readlines()]

        assert len(music_src) == len(music_trg)
        assert len(family_src) == len(family_trg)

        record_list_src = music_src + family_src
        record_list_trg = music_trg + family_trg

        train_index, valid_index, test_index = data_split_index(record_list_src)

        src_list['train'] = [record_list_src[i] for i in train_index]
        trg_list['train'] = [record_list_trg[i] for i in train_index]

        src_list['valid'] = [record_list_src[i] for i in valid_index]
        trg_list['valid'] = [record_list_trg[i] for i in valid_index]

        src_list['test'] = [record_list_src[i] for i in test_index]
        trg_list['test'] = [record_list_trg[i] for i in test_index]

    # WNC [Biased -> Neutral]

    if args.data_name == 'WNC':
        args.data_path = os.path.join(args.data_path,'bias_data')
        col_names = ['ID','src_tok','tgt_tok','src_raw','trg_raw','src_POS','trg_parse_tags']

        train_dat = pd.read_csv(os.path.join(args.data_path, 'WNC/biased.word.train'), 
                                sep='\t', names=col_names)
        valid_dat = pd.read_csv(os.path.join(args.data_path, 'WNC/biased.word.dev'),
                                sep='\t', names=col_names)
        test_dat = pd.read_csv(os.path.join(args.data_path, 'WNC/biased.word.test'),
                               sep='\t', names=col_names)

        src_list['train'] = train_dat['src_raw'].tolist()
        trg_list['train'] = train_dat['trg_raw'].tolist()
        src_list['valid'] = valid_dat['src_raw'].tolist()
        trg_list['valid'] = valid_dat['trg_raw'].tolist()
        src_list['test'] = test_dat['src_raw'].tolist()
        trg_list['test'] = test_dat['trg_raw'].tolist()

    #===================================#
    #==========Classification===========#
    #===================================#

    if args.data_name == 'korean_hate_speech':
        args.data_path = os.path.join(args.data_path,'korean-hate-speech-detection')

        train_dat = pd.read_csv(os.path.join(args.data_path, 'train.hate.csv'))
        valid_dat = pd.read_csv(os.path.join(args.data_path, 'dev.hate.csv'))
        test_dat = pd.read_csv(os.path.join(args.data_path, 'test.hate.no_label.csv'))

        train_dat['label'] = train_dat['label'].replace('none', 0)
        train_dat['label'] = train_dat['label'].replace('hate', 1)
        train_dat['label'] = train_dat['label'].replace('offensive', 2)
        valid_dat['label'] = valid_dat['label'].replace('none', 0)
        valid_dat['label'] = valid_dat['label'].replace('hate', 1)
        valid_dat['label'] = valid_dat['label'].replace('offensive', 2)

        src_list['train'] = train_dat['comments'].tolist()
        trg_list['train'] = train_dat['label'].tolist()
        src_list['valid'] = valid_dat['comments'].tolist()
        trg_list['valid'] = valid_dat['label'].tolist()
        src_list['test'] = test_dat['comments'].tolist()
        trg_list['test'] = [0 for _ in range(len(test_dat))]

    if args.data_name == 'IMDB':
        args.data_path = os.path.join(args.data_path,'text_classification/IMDB')

        if args.with_eda:
            train_dat = pd.read_csv(os.path.join(args.data_path, 'train_aug.csv'))
        else:
            train_dat = pd.read_csv(os.path.join(args.data_path, 'train.csv'))

        test_dat = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
        test_dat['sentiment'] = test_dat['sentiment'].replace('positive', 0)
        test_dat['sentiment'] = test_dat['sentiment'].replace('negative', 1)

        train_index, valid_index, test_index = data_split_index(train_dat)

        src_list['train'] = [train_dat['comment'].tolist()[i] for i in train_index]
        trg_list['train'] = [train_dat['sentiment'].tolist()[i] for i in train_index]

        src_list['valid'] = [train_dat['comment'].tolist()[i] for i in valid_index]
        trg_list['valid'] = [train_dat['sentiment'].tolist()[i] for i in valid_index]

        src_list['test'] = test_dat['comment'].tolist()
        trg_list['test'] = test_dat['sentiment'].tolist()

    if args.data_name == 'ProsCons':
        args.data_path = os.path.join(args.data_path,'text_classification/ProsCons')

        if args.with_eda:
            train_dat = pd.read_csv(os.path.join(args.data_path, 'train_aug.csv'))
        else:
            train_dat = pd.read_csv(os.path.join(args.data_path, 'train.csv'), names=['label', 'description'])

        test_dat = pd.read_csv(os.path.join(args.data_path, 'test.csv'), names=['label', 'description'])

        train_index, valid_index, test_index = data_split_index(train_dat)

        src_list['train'] = [train_dat['description'].tolist()[i] for i in train_index]
        trg_list['train'] = [train_dat['label'].tolist()[i] for i in train_index]

        src_list['valid'] = [train_dat['description'].tolist()[i] for i in valid_index]
        trg_list['valid'] = [train_dat['label'].tolist()[i] for i in valid_index]

        src_list['test'] = test_dat['description'].tolist()
        trg_list['test'] = test_dat['label'].tolist()

    if args.data_name == 'MR':
        args.data_path = os.path.join(args.data_path,'text_classification/MR')

        if args.with_eda:
            train_dat = pd.read_csv(os.path.join(args.data_path, 'train_aug.csv'))
        else:
            train_dat = pd.read_csv(os.path.join(args.data_path, 'train.csv'), names=['label', 'description'])

        test_dat = pd.read_csv(os.path.join(args.data_path, 'test.csv'), names=['label', 'description'])

        train_index, valid_index, test_index = data_split_index(train_dat)

        src_list['train'] = [train_dat['description'].tolist()[i] for i in train_index]
        trg_list['train'] = [train_dat['label'].tolist()[i] for i in train_index]

        src_list['valid'] = [train_dat['description'].tolist()[i] for i in valid_index]
        trg_list['valid'] = [train_dat['label'].tolist()[i] for i in valid_index]

        src_list['test'] = test_dat['description'].tolist()
        trg_list['test'] = test_dat['label'].tolist()

    if args.data_name == 'GVFC':
        args.data_path = os.path.join(args.data_path,'GVFC')

        gvfc_dat = pd.read_csv(os.path.join(args.data_path, 'GVFC_headlines_and_annotations.csv'))
        gvfc_dat = gvfc_dat.replace(99, 0)
        src_text = gvfc_dat['news_title'].tolist()
        trg_class = gvfc_dat['Q3 Theme1'].tolist()

        train_index, valid_index, test_index = data_split_index(gvfc_dat)

        src_list['train'] = [src_text[i] for i in train_index]
        trg_list['train'] = [trg_class[i] for i in train_index]
        src_list['valid'] = [src_text[i] for i in valid_index]
        trg_list['valid'] = [trg_class[i] for i in valid_index]
        src_list['test'] = [src_text[i] for i in test_index]
        trg_list['test'] = [trg_class[i] for i in test_index]

    #===================================#
    #===========Summarization===========#
    #===================================#

    if args.data_name == 'cnn_dailymail':

        args.data_path = os.path.join(args.data_path, 'cnn_dailymail', args.cnn_dailymail_ver)

        train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
        valid = pd.read_csv(os.path.join(args.data_path, 'valid.csv'))
        test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

        src_list['train'] = train['article'].tolist()
        src_list['valid'] = valid['article'].tolist()
        src_list['test'] = test['article'].tolist()

        trg_list['train'] = train['summary'].tolist()
        trg_list['valid'] = valid['summary'].tolist()
        trg_list['test'] = test['summary'].tolist()

    #===================================#
    #===============GLUE================#
    #===================================#
    
    if ''.join(args.data_name.split('_')[0]) == 'glue' or ''.join(args.data_name.split('_')[:2]) == 'superglue':
        if ''.join(args.data_name.split('_')[0]) == 'glue':
            task = '_'.join(args.data_name.split('_')[1:])
        else:
            task = args.data_name.split('_')[-1]
        
        dataset = load_dataset('_'.join(args.data_name.split('_')[:-1]), task)

        return dataset

    #===================================#
    #====Multi-Modal_Classification=====#
    #===================================#

    if args.data_name == 'dacon_kotour':

        args.data_path = os.path.join(args.data_path, 'dacon_kotour')

        train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
        test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

        # Image path processing
        train['img_path'] = train['img_path'].map(lambda t: os.path.join(args.data_path, 'image/train', t.split('/')[-1]))
        test['img_path'] = test['img_path'].map(lambda t: os.path.join(args.data_path, 'image/test', t.split('/')[-1]))
        
        train_index, valid_index, test_index = data_split_index(train, test_ratio=0)

        le = LabelEncoder()
        le.fit(train.iloc[train_index]['cat3'].values)
        train['cat3_encoded'] = le.transform(train['cat3'].values)

        src_list['txt'] = dict()
        src_list['img'] = dict()

        src_list['img']['train'] = [train['img_path'].tolist()[i] for i in train_index]
        src_list['txt']['train'] = [train['overview'].tolist()[i] for i in train_index]
        trg_list['train'] = [train['cat3_encoded'].tolist()[i] for i in train_index]

        src_list['img']['valid'] = [train['img_path'].tolist()[i] for i in valid_index]
        src_list['txt']['valid'] = [train['overview'].tolist()[i] for i in valid_index]
        trg_list['valid'] = [train['cat3_encoded'].tolist()[i] for i in valid_index]

        src_list['img']['test'] = train['img_path'].tolist()
        src_list['txt']['test'] = test['overview'].tolist()

    if args.src_trg_reverse:

        assert args.task != 'classification'

        src_list_copy = trg_list
        trg_list = src_list
        src_list = src_list_copy

    return src_list, trg_list

    ## Process
    # data_preprocess call data_load
    # data_load load dataset and return src, trg
    # data_preprocess preprocessing data and save processed data

    # args.data_name = ['glue', 'super_glue']

    # glue_list = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
    # super_glue_list = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg']

    ## Glue
    # cola : sentence, label
    # sst2 : sentence, label
    # mrpc : sentence1, sentence2, label
    # qqp : question1, question2, label
    # stsb : sentence1, sentence2, label
    # mnli : premise, hypothesis, label
    # mnli_mismatched(NO TRAIN DATASET!!!) : premise, hypothesis, label 
    # qnli : question, sentence, label
    # rte : sentence1, sentence2, label
    # wnli : sentence1, sentence2, label
    # ax(JUST TEST DATASET!!!) : premise, hypothesis, label

    ## Super Glue
    # boolq : question, passage, label
    # cb : premise, hypothesis, label
    # copa : premise, choice1, choice2, question, label
    # multirc : paragraph, question, answer, label / 여기 index 가 다른데?
    # record : passage, query, entities, answers
    # rte : premise, hypothesis, label
    # wic : word, sentence1, sentence2, start1, start2, end1, end2, label
    # wsc : text, span1_index, spna2_index, span1_text, span2_text, label
    # wsc.fixed : text, span1_index, spna2_index, span1_text, span2_text, label
    # axb(JUST TEST DATASET!!!) : sentence1, sentece2, label
    # axg(JUST TEST DATASET!!!) : premise, hypothesis, label
