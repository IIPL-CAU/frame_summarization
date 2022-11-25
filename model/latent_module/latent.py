# Import PyTorch
import torch
from torch import nn
from torch.autograd import Variable
# Import custom modules
from .encoder_decoder import full_cnn_latent_encoder, full_cnn_latent_decoder, cnn_latent_encoder, cnn_latent_decoder
from .loss import GaussianKLLoss, MaximumMeanDiscrepancyLoss

class Latent_module(nn.Module):
    def __init__(self, d_model: int = 512, d_latent: int = 256, variational_model: str = 'vae', 
                 variational_token_processing: str = 'average', variational_with_target: bool = False,
                 cnn_encoder: bool = False, cnn_decoder: bool = False, latent_add_encoder_out: bool = True, 
                 z_var: int = 2, src_max_len: int = 300, trg_max_len: int = 300):

        super(Latent_module, self).__init__()

        self.variational_model = variational_model
        self.variational_token_processing = variational_token_processing
        self.variational_with_target = variational_with_target
        self.latent_add_encoder_out = latent_add_encoder_out
        self.cnn_encoder = cnn_encoder
        self.cnn_decoder = cnn_decoder

        self.z_var = z_var
        self.loss_lambda = 1
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
        
        # Variational Autoencoder
        if self.variational_model == 'vae':
            if self.cnn_encoder:
                self.context_to_mu = nn.Linear(d_latent, d_latent)
                self.context_to_logvar = nn.Linear(d_latent, d_latent)
            else:
                self.context_to_mu = nn.Linear(d_model, d_latent)
                self.context_to_logvar = nn.Linear(d_model, d_latent)
            self.z_to_context = nn.Linear(d_latent, d_model)

        # Wasserstein Autoencoder
        if self.variational_model == 'wae':
            self.context_to_latent = nn.Linear(d_model, d_latent)
            self.latent_to_context = nn.Linear(d_latent, d_model)

            self.mmd_criterion = MaximumMeanDiscrepancyLoss()

        # CNN Encoder & Decoder
        if self.cnn_encoder:
            self.latent_encoder = full_cnn_latent_encoder(d_model, d_latent)
            if self.variational_model == 'vae':
                self.latent_to_mu = nn.Linear(d_latent, d_latent)
                self.latent_to_logvar = nn.Linear(d_latent, d_latent)
        if self.cnn_decoder:
            self.latent_decoder = full_cnn_latent_decoder(d_model, d_latent)
        # cnn 일반버젼 코딩도 진행해야함

    def forward(self, encoder_out_src, encoder_out_trg):

    #===================================#
    #================VAE================#
    #===================================#

        """
        1. Model dimension to latent dimenseion with 'context_to_mu' [seq_len, batch, d_latent]
        2. Average sequence token [batch, d_latent]
        3. Calculate Gaussian KL-Divergence
        4. Re-parameterization trick
        5. Decoding by 'z_to_context'
        """
        if self.variational_model == 'vae':

            # 1-1. Model dimension to latent dimenseion with CNN encoder
            if self.cnn_encoder:

                # Source encoding
                if self.src_max_len == 100:
                    src_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:-2](encoder_out_src) # [batch, d_latent]
                elif self.src_max_len == 300:
                    src_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:](encoder_out_src) # [batch, d_latent]
                elif self.src_max_len == 768:
                    src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
                else:
                    raise Exception('Sorry, Now only 100, 300, 768 length is available')

                src_mu = self.latent_to_mu(src_latent)
                src_logvar = self.latent_to_logvar(src_latent)

                # Target encoding
                if self.variational_with_target:
                    if self.trg_max_len == 100:
                        trg_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:-2](encoder_out_trg) # [batch, d_latent]
                    elif self.trg_max_len == 300:
                        trg_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:](encoder_out_trg) # [batch, d_latent]
                    elif self.trg_max_len == 768:
                        trg_latent = self.latent_encoder(encoder_out_trg) # [batch, d_latent]
                    else:
                        raise Exception('Sorry, Now only 100, 300, 768 length is available')

                    trg_mu = self.latent_to_mu(trg_latent)
                    trg_logvar = self.latent_to_logvar(trg_latent)

            # 1-2. Model dimension to latent dimenseion with 'context_to_mu'
            else:
                src_mu = self.context_to_mu(encoder_out_src) # [seq_len, batch, d_latent]
                src_logvar = self.context_to_logvar(encoder_out_src) # [seq_len, batch, d_latent]

                if self.variational_with_target:
                    trg_mu = self.context_to_mu(encoder_out_trg) # [seq_len, batch, d_latent]
                    trg_logvar = self.context_to_logvar(encoder_out_trg) # [seq_len, batch, d_latent]

                # 2. Sequence token processing
                if self.variational_token_processing == 'average':
                    src_mu = src_mu.mean(dim=0) # [batch, d_latent]
                    src_logvar = src_logvar.mean(dim=0) # [batch, d_latent]

                    if self.variational_with_target:
                        trg_mu = trg_mu.mean(dim=0) # [batch, d_latent]
                        trg_logvar = trg_logvar.mean(dim=0) # [batch, d_latent]

                if self.variational_token_processing == 'view':
                    batch_size = encoder_out_src.size(1)
                    src_mu = src_mu.view(batch_size, -1) # [batch, seq_len * d_latent]
                    src_logvar = src_logvar.view(batch_size, -1) # [batch, seq_len * d_latent]

                    if self.variational_with_target:
                        trg_mu = trg_mu.view(batch_size, -1) # [batch, seq_len * d_latent]
                        trg_logvar = trg_logvar.view(batch_size, -1) # [batch, seq_len * d_latent]

            # 3. Calculate Gaussian KL-Divergence
            if self.variational_with_target:
                numerator = src_logvar.exp() + torch.pow(src_mu - trg_mu, 2)
                fraction = torch.div(numerator, (trg_logvar.exp()))
                dist_loss = 0.5 * torch.sum(trg_logvar - src_logvar + fraction - 1, dim=1)
                dist_loss = dist_loss.mean() # Batch mean
            else:
                dist_loss = -0.5 * torch.sum(1 + src_logvar - src_mu.pow(2) - src_logvar.exp())

            # 4. Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu) # [batch, d_latent]

            # 5-1. Decoding by cnn
            if self.cnn_decoder:
                resize_z = self.latent_decoder(z) # [batch, seq_len, d_model]
            # 5-2. Decoding by 'z_to_context'
            else:
                resize_z = self.z_to_context(z) # [batch, d_model]
                resize_z = resize_z.unsqueeze(1).repeat(1, self.src_max_len, 1)

            # 6. Add latent variable or use only latent variable
            resize_z = resize_z.transpose(0, 1) # [seq_len, batch, d_model]
            if self.latent_add_encoder_out:
                encoder_out_total = torch.add(encoder_out_src, resize_z)
            else:
                encoder_out_total = resize_z

    #===================================#
    #================WAE================#
    #===================================#

        """
        1. Model dimension to latent dimenseion with 'context_to_latent' [seq_len, batch, d_latent]
        2. Average sequence token [batch, d_latent]
        3. Calculate Maximum-mean discrepancy
        4. Decoding with 'latent_to_context'
        """

        if self.variational_model == 'wae':

            # 1-1. Model dimension to latent dimenseion with CNN encoder
            if self.cnn_encoder:

                # Source encoding
                if self.src_max_len == 100:
                    src_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:-2](encoder_out_src) # [batch, d_latent]
                elif self.src_max_len == 300:
                    src_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:](encoder_out_src) # [batch, d_latent]
                elif self.src_max_len == 768:
                    src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
                else:
                    raise Exception('Sorry, Now only 100, 300, 768 length is available')

                # Target encoding
                if self.variational_with_target:
                    if self.src_max_len == 100:
                        trg_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:-2](encoder_out_src) # [batch, d_latent]
                    elif self.src_max_len == 300:
                        trg_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:](encoder_out_src) # [batch, d_latent]
                    elif self.src_max_len == 768:
                        trg_latent = self.latent_encoder(encoder_out_trg) # [batch, d_latent]
                    else:
                        raise Exception('Sorry, Now only 100, 300, 768 length is available')

            # 1-2. Model dimension to latent dimenseion
            else:
                src_latent = self.context_to_latent(encoder_out_src) # [seq_len, batch, d_latent]

                if self.variational_with_target:
                    trg_latent = self.context_to_latent(encoder_out_trg) # [seq_len, batch, d_latent]

            # 2. Sequence token processing
            if self.variational_token_processing == 'average':
                src_latent = src_latent.mean(dim=0) # [batch, d_latent]
                if self.variational_with_target:
                    trg_latent = trg_latent.mean(dim=0) # [batch, d_latent]

            if self.variational_token_processing == 'view':
                batch_size = encoder_out_src.size(1)
                src_latent = src_latent.view(batch_size, -1) # [batch, seq_len * d_latent]
                if self.variational_with_target:
                    trg_latent = trg_latent.view(batch_size, -1) # [batch, seq_len * d_latent]

            # 3. Calculate Maximum-mean discrepancy
            if self.variational_with_target:
                dist_loss = self.mmd_criterion(src_latent, trg_latent, self.z_var)
            else:
                sample_z = math.sqrt(self.z_var) * Variable(src_latent.data.new(src_latent.size()).normal_())
                dist_loss = self.mmd_criterion(src_latent, sample_z, self.z_var)

            # 4-1. Decoding by cnn
            if self.cnn_decoder:
                src_latent = self.latent_decoder(src_latent) # [batch, seq_len, d_model]
            # 4-2. Decoding by 'z_to_context'
            else:
                src_latent = self.latent_to_context(src_latent)

            # 5. Add latent variable or use only latent variable
            if self.latent_add_encoder_out:
                encoder_out_total = torch.add(encoder_out_src, src_latent)
            else:
                encoder_out_total = src_latent

        return encoder_out_total, dist_loss * 100 # Loss Lambda Refactoring 필수

    def generate(self, encoder_out_src):

    #===================================#
    #================VAE================#
    #===================================#

        if self.variational_model == 'vae':
            # 1-1. Model dimension to latent dimenseion with CNN encoder
            if self.cnn_encoder:

                # Source encoding
                if self.src_max_len == 100:
                    src_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:-2](encoder_out_src) # [batch, d_latent]
                elif self.src_max_len == 300:
                    src_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:](encoder_out_src) # [batch, d_latent]
                elif self.src_max_len == 768:
                    src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
                else:
                    raise Exception('Sorry, Now only 100, 300, 768 length is available')

                src_mu = self.latent_to_mu(src_latent)

            # 1-2. Model dimension to latent dimenseion with 'context_to_mu'
            else:
                src_mu = self.context_to_mu(encoder_out_src) # [token, batch, d_latent]

                # 2. Sequence token processing
                if self.variational_token_processing == 'average':
                    src_mu = src_mu.mean(dim=0) # [batch, d_latent]

                if self.variational_token_processing == 'view':
                    batch_size = encoder_out_src.size(1)
                    src_mu = src_mu.view(batch_size, -1) # [batch, seq_len * d_latent]

            # 3. Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu) # [batch, d_latent]

            # 5-1. Decoding by cnn
            if self.cnn_decoder:
                resize_z = self.latent_decoder(z) # [batch, seq_len, d_model]
            # 5-2. Decoding by 'z_to_context'
            else:
                resize_z = self.z_to_context(z) # [batch, d_model]
                resize_z = resize_z.unsqueeze(1).repeat(1, src_max_len, 1)

            # 6. Add latent variable or use only latent variable
            if self.latent_add_encoder_out:
                encoder_out_total = torch.add(encoder_out_src, resize_z)
            else:
                encoder_out_total = resize_z

    #===================================#
    #================WAE================#
    #===================================#

        if self.variational_model == 'wae':

            # 1-1. Model dimension to latent dimenseion with CNN encoder
            if self.cnn_encoder:

                # Source encoding
                if self.src_max_len == 100:
                    src_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:-2](encoder_out_src) # [batch, d_latent]
                elif self.src_max_len == 300:
                    src_latent = nn.Sequential(*list(self.latent_encoder.children()))[2:](encoder_out_src) # [batch, d_latent]
                elif self.src_max_len == 768:
                    src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
                else:
                    raise Exception('Sorry, Now only 100, 300, 768 length is available')
            
            else:
                # Source sentence latent mapping
                src_latent = self.context_to_latent(encoder_out_src) # [seq_len, batch, d_latent]

            # 2-1. Decoding by cnn
            if self.cnn_decoder:
                src_latent = self.latent_decoder(src_latent) # [seq_len, batch, d_model]
            # 2-2. Decoding by 'z_to_context'
            else:
                src_latent = self.latent_to_context(src_latent) # [seq_len, batch, d_model]

            # 3. Add latent variable or use only latent variable
            if self.latent_add_encoder_out:
                encoder_out_total = torch.add(encoder_out_src, src_latent)
            else:
                encoder_out_total = src_latent

    # #===================================#
    # #==============CNN+VAE==============#
    # #===================================#

    #     if self.variational_mode in [5,6]:

    #         src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
    #         src_mu = self.context_to_mu(src_latent) # [batch, d_latent]
            
    #         resize_z = self.latent_decoder(src_mu.unsqueeze(2)) # [seq_len, batch, d_model]

    #         encoder_out_total = torch.add(encoder_out_src, resize_z)
            
    # #===================================#
    # #==============CNN+WAE==============#
    # #===================================#

    #     if self.variational_mode == [7,8]:

    #         src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
    #         src_latent = self.latent_decoder(src_latent.unsqueeze(2)) # [token, batch, d_model]

    #         encoder_out_total = torch.add(encoder_out_src, src_latent)

    # #===================================#
    # #======Gaussian Mixture + VAE=======#
    # #===================================#

    #     """
    #     Need refactoring
    #     """

    #     if self.variational_mode == 9:

    #         # # 1. Style classifiction 
    #         # x_to_cls = encoder_out_src.mean(dim=0) # [batch, d_model]
    #         # distribution_cls = self.context_to_cls(x_to_cls) # [batch, num_cls]
    #         # distribution_cls_cp = distribution_cls.unsqueeze(0).repeat(encoder_out_src.size(0), 1, 1) # [token, batch, num_cls]

    #         # # concat
    #         # gm_inp = torch.cat((encoder_out_src, distribution_cls_cp), dim=2) # (tokn, batch, d_latent + num_cls)

    #         # src_mu = self.context_to_mu(gm_inp) # (token, batch, d_latent)
    #         # src_logvar = self.context_to_logvar(gm_inp) # (token, batch, d_latent)

    #         # mu = src_mu.view(gm_inp.size(1), -1) # [batch, seq_len * d_latent]
    #         # logvar = src_logvar.view(gm_inp.size(1), -1) # [batch, seq_len * d_latent]
    #         # dist_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #         # std = src_logvar.mul(0.5).exp_()
    #         # eps = Variable(std.data.new(std.size()).normal_())
    #         # z = eps.mul(std).add_(src_mu)

    #         # # 5. Decoding with 'z_to_context'
    #         # resize_z = self.z_to_context(z) # [batch, d_model]

    #         # encoder_out_total = torch.add(encoder_out_src, resize_z)

    #         raise Exception('Need refactoring...!! Try another variational mode')
            
    # #===================================#
    # #=========CNN+VAE(GMM&SIM)==========#
    # #===================================#

    #     if self.variational_mode == 10:
    #         # Source sentence latent mapping
    #         encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)

    #         # 1-1. Get content latent
    #         src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
    #         src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)

    #         # 1-2. VAE Process
    #         # 1-2-1. Get mu and logvar from src_content_latent and trg_content_latent
    #         src_content_mu = self.content_latent_to_mu(src_content_latent) # (batch_size, d_latent)
    #         src_content_logvar = self.content_latent_to_logvar(src_content_latent) # (batch_size, d_latent)

    #         # 1-2-2. Reparameterization trick
    #         src_content_std = src_content_logvar.mul(0.5).exp_()
    #         src_content_eps = torch.randn_like(src_content_std).to(self.device)
    #         src_content_z = src_content_eps * src_content_std + src_content_mu # (batch_size, d_latent)

    #         # 2-1. Get style latent
    #         src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
    #         src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)

    #         # 2-2. VAE Process
    #         # 2-2-1. Get mu and logvar from src_style_latent and trg_style_latent
    #         src_style_mu = self.style_latent_to_mu(src_style_latent) # (batch_size, d_latent)
    #         src_style_logvar = self.style_latent_to_logvar(src_style_latent) # (batch_size, d_latent)

    #         # 2-2-2. Reparameterization trick
    #         src_style_std = src_style_logvar.mul(0.5).exp_()
    #         src_style_eps = torch.randn_like(src_style_std).to(self.device)
    #         src_style_z = src_style_eps * src_style_std + src_style_mu # (batch_size, d_latent)

    #         # 3-1. Translate each src latent to d_model dimension
    #         src_content_latent = self.content_latent_decoder(src_content_z.unsqueeze(2)) # (batch_size, d_model, 1)
    #         src_style_latent = self.style_latent_decoder(src_style_z.unsqueeze(2)) # (batch_size, d_model, 1)

    #         # 3-2. add each src latent and repeat
    #         src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
    #         src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

    #         # 5. Get output
    #         encoder_out_total = torch.add(encoder_out_src, src_latent)
    #         encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)


    # #===================================#
    # #=========CNN+WAE(GMM&SIM)==========#
    # #===================================#
        
    #     if self.variational_mode == 11:
    #         # Source sentence latent mapping
    #         encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)

    #         # 1-1. Get content latent
    #         src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
    #         src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)

    #         # 2-1. Get style latent
    #         src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
    #         src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)
            
    #         # 3-1. Translate each src latent to d_model dimension
    #         src_content_latent = self.content_latent_decoder(src_content_latent.unsqueeze(2)) # (batch_size, d_model, 1)
    #         src_style_latent = self.style_latent_decoder(src_style_latent.unsqueeze(2)) # (batch_size, d_model, 1)

    #         # 3-2. add each src latent and repeat
    #         src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
    #         src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

    #         # 5. Get output
    #         encoder_out_total = torch.add(encoder_out_src, src_latent)
    #         encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)

    #     return encoder_out_total