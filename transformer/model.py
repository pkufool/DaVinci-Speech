import torch
import torch.nn as nn

from encoder import *
from decoder import *


class Transformer(nn.Module):
    def __init__(self, input_size, vocab_size, d_model=320, n_heads=4, d_ff=1280, num_enc_blocks=6, num_dec_blocks=6, residual_dropout_rate=0.1, share_embedding=True):
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.encoder = TransformerEncoder(input_size=input_size, d_model=d_model,
                                          attention_heads=n_heads,
                                          linear_units=d_ff,
                                          num_blocks=num_enc_blocks,
                                          residual_dropout_rate=residual_dropout_rate)

        self.decoder = TransformerDecoder(output_size=vocab_size,
                                          d_model=d_model,
                                          attention_heads=n_heads,
                                          linear_units=d_ff,
                                          num_blocks=num_dec_blocks,
                                          residual_dropout_rate=residual_dropout_rate,
                                          share_embedding=share_embedding)

        self.crit = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):

        # 1. forward encoder
        enc_states, enc_mask = self.encoder(inputs)

        # 2. forward decoder
        target_in = targets[:, :-1].clone()
        logits = self.decoder(target_in, enc_states, enc_mask)

        # 3. compute attention loss
        target_out = targets[:, 1:].clone()
        loss = self.crit(logits.reshape(-1, self.vocab_size), target_out.view(-1))

        return loss