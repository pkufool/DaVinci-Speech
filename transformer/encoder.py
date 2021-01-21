import torch
import torch.nn as nn

from component import *


class TransformerEncoderLayer(nn.Module):
    def __init__(self, attention_heads, d_model, linear_units, residual_dropout_rate):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)

    def forward(self, x, mask):
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        x = residual + self.dropout1(self.self_attn(x, x, x, mask))
        x = self.norm1(x)

        residual = x
        x = residual + self.dropout2(self.feed_forward(x))
        x = self.norm2(x)

        return x, mask


class TransformerEncoder(nn.Module):

    def __init__(self, input_size, d_model=256, attention_heads=4, linear_units=2048, num_blocks=6, 
                 repeat_times=1, pos_dropout_rate=0.0, slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0, 
                 residual_dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.embed = Conv2dSubsampling(input_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(attention_heads, d_model, linear_units, residual_dropout_rate) for _ in range(num_blocks)
        ])

    def forward(self, inputs):

        enc_mask = torch.sum(inputs, dim=-1).ne(0).unsqueeze(-2)
        enc_output, enc_mask = self.embed(inputs, enc_mask)

        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        for _, block in enumerate(self.blocks):
            enc_output, enc_mask = block(enc_output, enc_mask)
        
        return enc_output, enc_mask