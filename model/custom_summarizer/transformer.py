# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
# T5
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification
from model.custom_transformer.latent_module import Latent_module

class custom_BERT(nn.Module):
    def __init__(self, isPreTrain, d_latent, variational_mode, decoder_full_model, device):
        super().__init__()

        """

        """
        self.d_latent = d_latent
        self.isPreTrain = isPreTrain
        self.device = device

        # Frame Classifier
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if self.isPreTrain:
            self.frame_classifer = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        else:
            model_config = BertConfig('bert-base-uncased')
            self.frame_classifer = BertForSequenceClassification(config=model_config)

        # Extractive Summarizer
        if self.isPreTrain:
            self.ext_classifer = BertModel.from_pretrained('bert-base-uncased')
        else:
            model_config = BertConfig('bert-base-uncased')
            self.ext_classifer = BertModel(config=model_config)

        # VAE
        self.context_to_mu = nn.Linear(d_model, d_latent)
        self.context_to_logvar = nn.Linear(d_model, d_latent)

    def forward(self, src_input_ids, src_attention_mask,
                trg_input_ids, trg_attention_mask,
                non_pad_position=None, tgt_subsqeunt_mask=None):

        # Frame Classification
        # frame_out = self.frame_classifer(input_ids=src_input_ids,
        #                                  attention_mask=src_attention_mask)

        with torch.no_grad():
            encoder_out_trg = self.ext_classifer()

        src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
        src_logvar = self.context_to_logvar(encoder_out_src) # (token, batch, d_latent)

        trg_mu = self.context_to_mu(encoder_out_trg) # (token, batch, d_latent)
        trg_logvar = self.context_to_logvar(encoder_out_trg) # (token, batch, d_latent)

        numerator = src_logvar.exp() + torch.pow(src_mu - trg_mu, 2)
        fraction = torch.div(numerator, (trg_logvar.exp()))
        dist_loss = 0.5 * torch.sum(trg_logvar - src_logvar + fraction - 1, dim=0)

        return dist_loss