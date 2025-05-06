import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader
import numpy as np
from inference import SpikeFeatureExtractor
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import os

def check_for_nans(name, tensor):
    if torch.isnan(tensor).any():
        print(f"🚨 NaNs detected in {name}")
        return True
    return False

class Embedder(nn.Module):
    def __init__(self, d_encoder, dropout, should_log, max_len=1000):
        super(Embedder, self).__init__()
        self.d_encoder = d_encoder
        self.should_log = should_log
        self.max_len = max_len
        self.linear = nn.Linear(2, d_encoder)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = self.get_sin_cos_positional_encoding(max_len, d_encoder)
        self.register_buffer('pe', self.positional_encoding)
        self.register_buffer('mu', torch.zeros(2))
        self.register_buffer('sigma', torch.ones(2))

    def init_mu_sigma(self, mu, sigma):
        """
        :param mu: mean of the training set
        :param sigma: standard deviation of the training set
        """

        # on the right device ?
        mu = mu.to(self.mu.device)
        sigma = sigma.to(self.sigma.device)
        # check the size
        if mu.size() != (2,):
            raise ValueError(f"mu should be of size (2,), but got {mu.size()}")
        if sigma.size() != (2,):
            raise ValueError(f"sigma should be of size (2,), but got {sigma.size()}")
        
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)

    def get_sin_cos_positional_encoding(self, max_len, d_model):
        """
        :param max_len: maximum length of the sequence
        :param d_model: dimension of the model
        :return: positional encoding matrix of shape (max_len, d_model)
        """
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x, L):
        """
        :param x: input sequence of shape (batch_size, seq_len)
        :param L: length of the sequences before padding of shape (batch_size,)
        :return: embedded sequence of shape (batch_size, seq_len, d_encoder) and mask of shape (batch_size, seq_len - 1)
        """

        # == from x to x_features ==
        # x is the sequence of spike times
        # L is the length of the sequences before padding
        # we compute the ISIs
        if not self.should_log:
            x_ISI = torch.diff(x, dim=1)
        else:
            x_ISI = torch.log1p(torch.abs(torch.diff(x, dim=1)))
        L = L - 1
        # we compute the delta ISIs
        delta_ISI = torch.diff(x_ISI, dim=1)
        # we pad delta_ISI with one zero at the beginning
        delta_ISI = F.pad(delta_ISI, (1, 0), value=0)
        # we stack x_ISI and delta_ISI
        x_features = torch.stack((x_ISI, delta_ISI), dim=-1)
        # we normalize the features
        x_features = (x_features - self.mu) / self.sigma

        # == from x_features to z_emb ==

        # we apply the linear layer
        x_features = self.linear(x_features)
        # we add the positional encoding
        x_features = x_features + self.pe[:x_features.size(1), :]
        # we apply the dropout
        x_features = self.dropout(x_features)
        # we create the mask
        mask = torch.arange(x_features.size(1)).unsqueeze(0) < L.unsqueeze(1)
        mask = mask.to(x_features.device)
        # we apply the mask
        x_features = x_features * mask.unsqueeze(-1).float()
        # we return the embedded sequence and the mask

        # we check for NaNs
        if check_for_nans("x_features", x_features):
            raise ValueError("NaNs detected in x_features")

        return x_features, mask
    
class InteractionCore(nn.Module):
    def __init__(self, d_encoder, n_heads, dropout, n_blocks, activation):
        super(InteractionCore, self).__init__()

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()

        self.d_encoder = d_encoder
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.activation = activation

        # Define the multi-head attention and feed-forward layers
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(d_encoder, n_heads, dropout=dropout, batch_first=True) for _ in range(n_blocks)])
        self.feed_forward_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_encoder, d_encoder),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_encoder, d_encoder)
        ) for _ in range(n_blocks)])
        # Define layer normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_encoder) for _ in range(n_blocks)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_encoder) for _ in range(n_blocks)])
        # Define dropout layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_blocks)])
        # Define activation function
        self.activation_fn = activation
        # Define the final linear layer
        self.final_linear = nn.Linear(d_encoder, d_encoder)
        # Define the final layer normalization
        self.final_layer_norm = nn.LayerNorm(d_encoder)
        # Define the final dropout layer
        self.final_dropout = nn.Dropout(dropout)
        # Define the final activation function
        self.final_activation = activation

    def forward(self, x, mask):
        """
        :param x: input sequence of shape (batch_size, seq_len, d_encoder)
        :param mask: attention mask of shape (batch_size, seq_len)
        :return: processed sequence of shape (batch_size, seq_len, d_encoder) and mask of shape (batch_size, seq_len)
        """

        # == from z_emb to z_core ==
        # x is the embedded sequence
        # mask is the attention mask

        for i in range(self.n_blocks):
            # Multi-head attention
            attn_output, _ = self.attention_layers[i](x, x, x, key_padding_mask=~mask)
                
            # check for NaNs
            # it comes from here. Check why by inspecting the mask
            if check_for_nans("attn_output", attn_output):
                raise ValueError("NaNs detected in attn_output")

            x = x + self.dropout_layers[i](attn_output)
            x = self.layer_norms1[i](x)

            # Feed-forward network
            ff_output = self.feed_forward_layers[i](x)
            ff_output = self.activation_fn(ff_output)

            x = x + self.dropout_layers[i](ff_output)
            x = self.layer_norms2[i](x)
            # check for NaNs
            if check_for_nans("ff_output", ff_output):
                raise ValueError("NaNs detected in ff_output")
            if check_for_nans("x", x):
                raise ValueError("NaNs detected in x attention core")

        # Final linear layer
        x = self.final_linear(x)
        # Final layer normalization
        x = self.final_layer_norm(x)
        # Final dropout layer
        x = self.final_dropout(x)
        # Final activation function
        x = self.final_activation(x)
        # Return the processed sequence and mask

        #if check_for_nans("x", x):
        #    raise ValueError("NaNs detected in x after interaction core")

        return x, mask
    
class Pooler(nn.Module):
    def __init__(self, d_encoder, d_latent, dropout, activation):
        super(Pooler, self).__init__()

        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'gelu':
            activation_fn = nn.GELU()
        elif activation == 'silu':
            activation_fn = nn.SiLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.d_encoder = d_encoder
        self.d_latent = d_latent
        self.dropout = dropout
        self.activation_fn = activation_fn

        self.W_K = nn.Linear(d_encoder, d_encoder)
        self.W_pool = nn.Linear(d_encoder, d_latent)

        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_encoder)
        self.softmax = nn.Softmax(dim=-1)

        # Learnable query vector (shared across all batches)
        self.query_vector = nn.Parameter(torch.randn(1, 1, d_encoder))

    def forward(self, x, mask):
        """
        :param x: Tensor of shape (batch_size, seq_len, d_encoder)
        :param mask: Bool tensor of shape (batch_size, seq_len), where True means valid token
        :return: Tensor of shape (batch_size, d_latent), and original mask
        """

        batch_size, seq_len, _ = x.size()

        # Compute keys from input
        K = self.W_K(x)  # (B, T, d_encoder)

        # Expand query vector to match batch size
        Q = self.query_vector.expand(batch_size, -1, -1)  # (B, 1, d_encoder)

        # Attention scores (B, 1, T)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_encoder ** 0.5)
        with torch.cuda.amp.autocast(enabled=False):
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            attn_weights = self.softmax(attn_scores.float()).to(attn_scores.dtype)

        # Check for NaNs
        #if check_for_nans("attn_weights", attn_weights):
        #    raise ValueError("NaNs detected in attn_weights")

        # Weighted sum of values (pooled representation)
        pooled_representation = torch.bmm(attn_weights, x)  # (B, 1, d_encoder)
        pooled_representation = pooled_representation.squeeze(1)  # (B, d_encoder)

        # Normalize, project, and activate
        pooled_representation = self.layer_norm(pooled_representation)
        pooled_representation = self.W_pool(pooled_representation)
        pooled_representation = self.dropout_layer(pooled_representation)
        pooled_representation = self.activation_fn(pooled_representation)

        # Check for NaNs
        #if check_for_nans("pooled_representation", pooled_representation):
        #    raise ValueError("NaNs detected in pooled_representation")

        return pooled_representation, mask
    
class Encoder(nn.Module):
    def __init__(self, d_encoder, n_heads, dropout, n_blocks, d_latent, activation, should_log):
        super(Encoder, self).__init__()
        self.embedder = Embedder(d_encoder, dropout, should_log)
        self.interaction_core = InteractionCore(d_encoder, n_heads, dropout, n_blocks, activation)
        self.pooler = Pooler(d_encoder, d_latent, dropout, activation)
        
    def init_mu_sigma(self, mu, sigma):
        """
        :param mu: mean of the training set
        :param sigma: standard deviation of the training set
        """
        self.embedder.init_mu_sigma(mu, sigma)

    def forward(self, x, L):
        """
        :param x: input sequence of shape (batch_size, seq_len)
        :param L: length of the sequences before padding of shape (batch_size,)
        :return: pooled sequence of shape (batch_size, d_latent)
        """
        # Embed the input sequence
        x_emb, mask = self.embedder(x, L)

        # Process the embedded sequence through the interaction core
        x_core, mask = self.interaction_core(x_emb, mask)

        # Pool the processed sequence to obtain a fixed-size representation
        x_latent, mask = self.pooler(x_core, mask)

        return x_latent
    
class Decoder(nn.Module):
    def __init__(self, d_latent, dropout, n_blocks, activation, inference_only=False):
        super(Decoder, self).__init__()

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'silu':
            activation = nn.SiLU()
        elif activation == 'tanh':
            activation = nn.Tanh()

        self.d_latent = d_latent
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.activation = activation

        # Define the residual feedforward networks
        self.ff_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_latent, d_latent),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(d_latent, d_latent)
        ) for _ in range(n_blocks)])
        # Define layer normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_blocks)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_blocks)])
        # Define dropout layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_blocks)])
        # Define the final linear layer
        self.W_final = nn.Linear(d_latent, 2)

        # AUXILIARY
        if not inference_only:
            # Define the auxiliary head for classification
            self.classification_head = nn.ModuleList([nn.Sequential(
                nn.Linear(d_latent, d_latent),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(d_latent, d_latent)
            ) for _ in range(2)])
            # Define layer normalization for the classification head
            self.classification_layer_norms = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(2)])
            # Define dropout layers for the classification head
            self.classification_dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
            # Define the final linear layer for the classification head
            self.classification_final = nn.Linear(d_latent, 2)

            # Define the auxiliary head for metrics
            self.metrics_head = nn.ModuleList([nn.Sequential(
                nn.Linear(d_latent, d_latent),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(d_latent, d_latent)
            ) for _ in range(2)])
            # Define layer normalization for the metrics head
            self.metrics_layer_norms = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(2)])
            # Define dropout layers for the metrics head
            self.metrics_dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
            # Define the final linear layer for the metrics head
            self.metrics_final = nn.Linear(d_latent, 5)
            # Define the auxiliary head for uncertainty
            self.uncertainty_head = nn.ModuleList([nn.Sequential(
                nn.Linear(d_latent, d_latent),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(d_latent, d_latent)
            ) for _ in range(2)])
            # Define layer normalization for the uncertainty head
            self.uncertainty_layer_norms = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(2)])
            # Define dropout layers for the uncertainty head
            self.uncertainty_dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
            # Define the final linear layer for the uncertainty head
            self.uncertainty_final = nn.Linear(d_latent, 2)

            # INITIALIZE THE UNCERTAINTY HEAD WITH VERY SMALL VALUES
            self.uncertainty_final.weight.data *= 0.01

        else:
            # In inference mode, we don't need the auxiliary heads
            self.classification_head = None
            self.metrics_head = None
            self.uncertainty_head = None

            print("Inference mode: Auxiliary heads are not used.")

        # two vectors of parameters of size d_latent
        self.blending_vector_c = nn.Parameter(torch.randn(d_latent))
        self.blending_vector_m = nn.Parameter(torch.randn(d_latent))

    def predict(self, x):
        raise NotImplementedError("Please use forward_auxilliary method for training and predict for inference")
    
    def forward(self, x):
        #raise an error, we should use either predict or forward_auxilliary
        raise NotImplementedError("Please use predict method for inference and forward_auxilliary for training")

    def forward_auxilliary(self, x):
        """
        :param x: input sequence of shape (batch_size, d_latent)
        :return: output sequence of shape (batch_size, 2) and auxiliary outputs
        """

        x_temp = x

        # == from z_latent to y_hat_aux ==
        # we compute the classification head
        if self.classification_head is not None:
            x_aux_c = x_temp
            for i in range(2):
                ff_output = self.classification_head[i](x_aux_c)
                ff_output = self.activation(ff_output)
                x_aux_c = x_aux_c + self.classification_dropout_layers[i](ff_output)
                x_aux_c = self.classification_layer_norms[i](x_aux_c)

            y_hat_aux_c = self.classification_final(x_aux_c)

            x_aux_m = x_temp
            # we compute the metrics head
            for i in range(2):
                ff_output = self.metrics_head[i](x_aux_m)
                ff_output = self.activation(ff_output)
                x_aux_m = x_aux_m + self.metrics_dropout_layers[i](ff_output)
                x_aux_m = self.metrics_layer_norms[i](x_aux_m)

            y_hat_aux_m = self.metrics_final(x_aux_m)

            # == from z_latent to y_hat ==
            # x is the fixed-size representation
            x_aux = x_temp + self.blending_vector_c * x_aux_c + self.blending_vector_m * x_aux_m
            for i in range(self.n_blocks):
                # Feed-forward network
                ff_output = self.ff_layers[i](x_aux)
                ff_output = self.activation(ff_output)
                x_aux = x_aux + self.dropout_layers[i](ff_output)
                x_aux = self.layer_norms1[i](x_aux)

            # check
            #if check_for_nans("x", x):
            #    raise ValueError("NaNs detected in x in the decoder")

            # Final linear layer
            y_hat = self.W_final(x_aux)

            # we compute the uncertainty head
            for i in range(2):
                ff_output = self.uncertainty_head[i](x_aux)
                ff_output = self.activation(ff_output)
                x_aux = x_aux + self.uncertainty_dropout_layers[i](ff_output)
                x_aux = self.uncertainty_layer_norms[i](x_aux)
            y_hat_aux_s = self.uncertainty_final(x_aux)

            # check
            #if check_for_nans("y_hat_aux_c", y_hat_aux_c):
            #    raise ValueError("NaNs detected in y_hat_aux_c in the decoder")
            #if check_for_nans("y_hat_aux_m", y_hat_aux_m):
            #    raise ValueError("NaNs detected in y_hat_aux_m in the decoder")
            #if check_for_nans("y_hat_aux_s", y_hat_aux_s):
            #    raise ValueError("NaNs detected in y_hat_aux_s in the decoder")
            # check
            #if check_for_nans("y_hat", y_hat):
            #    raise ValueError("NaNs detected in y_hat in the decoder")

            return y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s

        return y_hat
        
class DICsNet(nn.Module):
    def __init__(self, d_encoder, n_heads, dropout, n_blocks_encoder, n_blocks_decoder, d_latent, activation, inference_only=False, should_log=False):
        super(DICsNet, self).__init__()
        self.encoder = Encoder(d_encoder, n_heads, dropout, n_blocks_encoder, d_latent, activation, should_log)
        self.decoder = Decoder(d_latent, dropout, n_blocks_decoder, activation, inference_only=inference_only)

    def init_mu_sigma(self, mu, sigma):
        """
        :param mu: mean of the training set
        :param sigma: standard deviation of the training set
        """
        self.encoder.embedder.init_mu_sigma(mu, sigma)

    def predict(self, x, L):
        """
        :param x: input sequence of shape (batch_size, seq_len)
        :param L: length of the sequences before padding of shape (batch_size,)
        :return: output sequence of shape (batch_size, 2)
        """
        # Forward pass through the encoder
        x_latent = self.encoder(x, L)
        # Forward pass through the decoder
        y_hat = self.decoder.predict(x_latent)
        return y_hat
    def forward(self, x):
        raise NotImplementedError("Please use predict method for inference and forward_auxilliary for training")
    def forward_auxilliary(self, x, L):
        """
        :param x: input sequence of shape (batch_size, seq_len)
        :param L: length of the sequences before padding of shape (batch_size,)
        :return: output sequence of shape (batch_size, 2) and auxiliary outputs
        """
        # Forward pass through the encoder
        x_latent = self.encoder(x, L)

        # check for NaNs
        if check_for_nans("x_latent", x_latent):
            raise ValueError("NaNs detected in x_latent")

        # Forward pass through the decoder
        y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s = self.decoder.forward_auxilliary(x_latent)
        
        return y_hat, y_hat_aux_c, y_hat_aux_m, y_hat_aux_s
    
    @staticmethod
    def HeteroscedasticHuberLoss(y, y_hat, log_sigma):
        """
        :param y: ground truth values
        :param y_hat: predicted values
        :param log_sigma: log of predicted uncertainties
        :return: heteroscedastic Huber loss
        """
        delta = 1.0

        #clamp
        log_sigma = torch.clamp(log_sigma, min=-5, max=2)

        # Compute sigma from log_sigma
        sigma = torch.exp(log_sigma)

        if torch.any(torch.isnan(sigma)):
            raise ValueError("NaN values detected in sigma")

        # Compute the Huber loss
        huber_loss = torch.where(torch.abs(y - y_hat) < delta,
                                0.5 * (y - y_hat) ** 2,
                                delta * (torch.abs(y - y_hat) - 0.5 * delta))
        # Compute the heteroscedastic loss
        heteroscedastic_loss = huber_loss / (sigma ** 2 + 1e-6) + 2 * log_sigma
        return heteroscedastic_loss.mean()
