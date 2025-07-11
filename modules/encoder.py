import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    """
    Encoder Layer의 구조

    Encoder_input = embedded_input + positional_encoding
    EncoderLayer 구조:
    [Encoder_input] -> [MultiHeadAttention(input, head)] -> [Add & Norm] -> [FFN] -> [Add & Norm]

    Args:
        input_size: 입력 및 출력의 차원 (hidden size)
        num_head: Multi-Head Attention의 head 개수
        ff_size: FFN 내부의 중간 차원

    forward Args:
        x (Tensor): [batch_size, input_length, input_size]
        mask (Tensor): padding_mask = [batch_size, input_length]

    Returns:
        output (Tensor): [batch_size input_length, input_size]
    """
    def __init__(self, input_size, num_heads, ff_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(input_size)
        self.ffn = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, input_size)
        )
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x, mask):
        # MultiHeadAttention
        attn_ouput, _ = self.attention(x, x, x, key_padding_mask=mask) # q, k, v = x
        x = self.norm1(x + attn_ouput)

        #Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class Encoder(nn.Module):
    """
    Transformer Encoder main

    구조:
    [EncoderLayer] * num_layers -> LayerNorm

    Args:
        num_layers (int): Encoder Layer의 반복 횟수
        input_size (int): 입력/출력 벡터의 차원 (hidden size)
        num_heads (int): 멀티헤드 어텐션의 헤드 개수
        ff_size (int): FFN 내부 차원

    forward Args:
        x (Tensor): [batch_size, input_length, input_size]
        mask (Tensor): padding_mask = [batch_size, input_length]

    Returns:
        output (Tensor): [batch_size input_length, input_size]
    """
    def __init__(self, num_layers, input_size, num_heads, ff_size):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(input_size, num_heads, ff_size) for _ in range(num_layers)
            ])
        self.norm = nn.LayerNorm(input_size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)