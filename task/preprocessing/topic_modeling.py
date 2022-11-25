import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import nltk
import torch
import logging
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.mixture import GaussianMixture
from nltk.corpus import stopwords as stop_words
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords as sp

from task.preprocessing.data_load import total_data_load
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name

def topic_modeling(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, 'Data Loading...')

    nltk.download('stopwords')
    stopwords = list(stop_words.words("english"))

    src_list, trg_list = total_data_load(args)

    #===================================#
    #==========Topic Modeling===========#
    #===================================#

    write_log(logger, 'Start preprocessing...')

    if args.topic_modeling_model == 'ctm':

        sp_train_out = sp(src_list['train'], stopwords_list=stopwords).preprocess()
        sp_valid_out = sp(src_list['valid'], stopwords_list=stopwords).preprocess()
        sp_test_out = sp(src_list['test'], stopwords_list=stopwords).preprocess()

        qt = TopicModelDataPreparation("all-mpnet-base-v2")
        train_dataset = qt.fit(text_for_contextual=sp_train_out[0], text_for_bow=sp_train_out[1])
        valid_dataset = qt.transform(text_for_contextual=sp_valid_out[0], text_for_bow=sp_valid_out[1])
        test_dataset = qt.transform(text_for_contextual=sp_test_out[0], text_for_bow=sp_test_out[1])

        write_log(logger, 'Start topc-model training...')

        ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=args.n_components, num_epochs=args.topic_epochs) # 7 topics
        ctm.fit(train_dataset) # run the model
        ctm.save(models_dir=args.preprocess_path) # save the model

        topic_keywords_list = ctm.get_topic_lists(7)
        for i in range(args.n_components):
            write_log(logger, topic_keywords_list[i])

        pd.DataFrame(topic_keywords_list).to_csv(f'./preprocessed/{args.n_components}_{args.topic_epochs}.csv', index=False)

    elif args.topic_modeling_model == 'bert_gmm':
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distilbert-base-nli-mean-tokens')
        
        write_log(logger, 'Start BoW Making...')

        vector = CountVectorizer(vocabulary=tokenizer.vocab)
        bow_array = vector.fit_transform(src_list['train']).toarray()

        write_log(logger, 'Start S-BERT...')

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = model.encode(src_list['train'], show_progress_bar=True)

        write_log(logger, 'Start Concatenate...')

        embeddings_cat = np.concatenate((embeddings, bow_array * args.umap_bow_lambda), axis=1)

        write_log(logger, 'Start UMAP...')

        umap_embeddings = umap.UMAP(n_neighbors=args.umap_n_neighbors, 
                                    n_components=args.umap_n_components, 
                                    metric='cosine').fit_transform(embeddings)

        write_log(logger, 'Start GMM...')

        gm = GaussianMixture(n_components=args.n_components, random_state=0)
        gm.fit(umap_embeddings)

        docs_df = pd.DataFrame(src_list['train'], columns=["Doc"])
        docs_df['Topic'] = gm.predict(umap_embeddings)
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

        def c_tf_idf(documents, m, ngram_range=(1, 1)):
            count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
            t = count.transform(documents).toarray()
            w = t.sum(axis=1)
            tf = np.divide(t.T, w)
            sum_t = t.sum(axis=0)
            idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
            tf_idf = np.multiply(tf, idf)

            return tf_idf, count

        def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
            words = count.get_feature_names()
            labels = list(docs_per_topic.Topic)
            tf_idf_transposed = tf_idf.T
            indices = tf_idf_transposed.argsort()[:, -n:]
            top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
            return top_n_words

        def extract_topic_sizes(df):
            topic_sizes = (df.groupby(['Topic'])
                             .Doc
                             .count()
                             .reset_index()
                             .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                             .sort_values("Size", ascending=False))
            return topic_sizes

        write_log(logger, 'Start processing...')

        tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(src_list['train']))
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
        topic_sizes = extract_topic_sizes(docs_df)

        topic_dict = dict()
        for i in range(args.n_components):
            write_log(logger, top_n_words[i][:10])
            topic_dict[f'{i}'] = list()
            for q in top_n_words[i][:10]:
                topic_dict[f'{i}'].append(q[0])

        pd.DataFrame(topic_dict).to_csv(f'./preprocessed/bertopic_gmm_{args.n_components}_{args.topic_epochs}.csv', index=False)
        docs_df.to_csv(f'./preprocessed/bertopic_gmm_results_{args.n_components}_{args.topic_epochs}.csv', index=False)