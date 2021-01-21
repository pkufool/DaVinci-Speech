import torch
import torch.nn as nn

from component import *


class TransformerDecoderLayer(nn.Module):

    def __init__(self, attention_heads, d_model, linear_units, residual_dropout_rate):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model)
        self.src_attn = MultiHeadedAttention(attention_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features

        :param torch.Tensor tgt: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        :param torch.Tensor memory: encoded source features (batch, max_time_in, size)
        :param torch.Tensor memory_mask: mask for memory (batch, max_time_in)
        """
        residual = tgt
        x = residual + self.dropout1(self.self_attn(tgt, tgt, tgt, tgt_mask))
        x = self.norm1(x)

        residual = x
        x = residual + self.dropout2(self.src_attn(x, memory, memory, memory_mask))
        x = self.norm2(x)

        residual = x
        x = residual + self.dropout3(self.feed_forward(x))
        x = self.norm3(x)

        return x, tgt_mask


def get_seq_mask(targets):
    batch_size, steps = targets.size()
    seq_mask = torch.ones([batch_size, steps, steps], device=targets.device)
    seq_mask = torch.tril(seq_mask).bool()
    return seq_mask


class TransformerDecoder(nn.Module):
    def __init__(self, output_size, d_model=256, attention_heads=4, linear_units=2048, num_blocks=6,     
                 residual_dropout_rate=0.1, share_embedding=False):
        super(TransformerDecoder, self).__init__()

        self.embedding = torch.nn.Embedding(output_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.blocks = nn.ModuleList([
            TransformerDecoderLayer(attention_heads, d_model, linear_units, 
                                    residual_dropout_rate) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(d_model, output_size)

        if share_embedding:
            assert self.embedding.weight.size() == self.output_layer.weight.size()
            self.output_layer.weight = self.embedding.weight

    def forward(self, targets, memory, memory_mask):

        dec_output = self.embedding(targets)
        dec_output = self.pos_encoding(dec_output)

        dec_mask = get_seq_mask(targets)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        logits = self.output_layer(dec_output)

        return logits

    def recognize(self, preds, memory, memory_mask, last=True):

        dec_output = self.embedding(preds)
        dec_mask = get_seq_mask(preds)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        logits = self.output_layer(dec_output)

        log_probs = F.log_softmax(logits[:, -1] if last else logits, dim=-1)

        return log_probs