import math

import torch
import torch.nn as nn

NEG_INF = -10000.0


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def form_attention_mask(input_mask, diagonal=None):
    """
    Build attention mask with optional masking of future tokens we forbid
    to attend to (e.g. as it is in Transformer decoder).

    Args:
        input_mask: binary mask of size B x L with 1s corresponding to valid
            tokens and 0s corresponding to padding tokens
        diagonal: diagonal where triangular future mask starts
            None -- do not mask anything
            0 -- regular translation or language modeling future masking
            1 -- query stream masking as in XLNet architecture
    Returns:
        attention_mask: mask of size B x 1 x L x L with 0s corresponding to
            tokens we plan to attend to and -10000 otherwise
    """

    if input_mask is None:
        return None
    attn_shape = (1, input_mask.shape[1], input_mask.shape[1])
    attn_mask = input_mask.byte().unsqueeze(1)
    if diagonal is not None:
        future_mask = torch.tril(
            torch.ones(attn_shape).byte().to(input_mask.device), diagonal)
        attn_mask = attn_mask & future_mask
    attention_mask = (1 - attn_mask.to(torch.float)) * NEG_INF
    return attention_mask.unsqueeze(1)


def transformer_weights_init(module, std_init_range=0.02, xavier=True):
    """
    Initialize different weights in Transformer model.

    Args:
        module: torch.nn.Module to be initialized
        std_init_range: standard deviation of normal initializer
        xavier: if True, xavier initializer will be used in Linear layers
            as was proposed in AIAYN paper, otherwise normal initializer
            will be used (like in BERT paper)
    """

    if isinstance(module, nn.Linear):
        if xavier:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
