import torch
import torch.nn as nn

def create_causal_mask(size):
    """미래 토큰을 가리는 causal mask
    
    Args:
        size (int): 시퀀스 길이(토큰 개수)

    Returns:
        torch.Tensor: [size, size] 크기의 bool tensor
                      - 0(False) : 보이는 부분
                      - 1(True)  : masked(O)    
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

def create_padding_mask(tokens, pad_token_id):
    """
    Padding Mask 생성
    
    Args:
        tokens (Tensor): [batch_size, seq_len] - 토큰 인덱스들
        pad_token_id (int): 패딩 토큰의 ID
        
    Returns:
        mask (Tensor): [batch_size, seq_len] - True인 곳은 패딩 토큰
    """
    return (tokens == pad_token_id)

def create_decoder_masks(src_tokens, tgt_tokens, pad_token_id):
    """
    Decoder에서 사용할 모든 마스크들을 생성
    
    Args:
        src_tokens (Tensor): [batch_size, src_len] - 소스 토큰들
        tgt_tokens (Tensor): [batch_size, tgt_len] - 타겟 토큰들  
        pad_token_id (int): 패딩 토큰 ID
        
    Returns:
        target_mask (Tensor): [tgt_len, tgt_len] - Causal mask
        memory_mask (Tensor): [batch_size, src_len] - Encoder padding mask
    """
    tgt_len = tgt_tokens.size(1)
    
    target_mask = create_causal_mask(tgt_len)
    target_mask = target_mask.to(tgt_tokens.device)

    memory_mask = create_padding_mask(src_tokens, pad_token_id)
    
    return target_mask, memory_mask



class DecoderLayer(nn.Module):
    """
    Decoder Layer의 구조
    
    DecoderLayer 구조:
    [Decoder_input] 
    -> [Masked Multi-Head Self-Attention + Add & Norm] 
    -> [Multi-Head Cross-Attention (Encoder-Decoder) + Add & Norm] 
    -> [FFN + Add & Norm]
    
    Args:
        input_size: 입력 및 출력의 차원 (hidden size)
        num_heads: Multi-Head Attention의 head 개수
        ff_size: FFN 내부의 중간 차원
        dropout: 드롭아웃 확률
        
    forward Args:
        x (Tensor): [batch_size, target_length, input_size] - Decoder 입력
        encoder_output (Tensor): [batch_size, source_length, input_size] - Encoder 출력
        target_mask (Tensor): [target_length, target_length] - 미래 토큰 마스킹
        memory_mask (Tensor): [batch_size, source_length] - Encoder padding 마스크
        
    Returns:
        output (Tensor): [batch_size, target_length, input_size]
    """
    def __init__(self, input_size, num_heads, ff_size, dropout=0.1):
        super().__init__()
        
        # Masked Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_size)
        self.dropout1 = nn.Dropout(dropout)
        
        # Multi-Head Cross-Attention (Encoder-Decoder Attention)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size)
        )
        self.norm3 = nn.LayerNorm(input_size)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, target_mask=None, memory_mask=None):
        # Masked Multi-Head Self-Attention
        self_attn_output, _ = self.self_attention(
            x, x, x, 
            attn_mask=target_mask,      # 미래 토큰 마스킹
            key_padding_mask=None       
        )
        x = self.norm1(x + self.dropout1(self_attn_output))  
        
        # Multi-Head Cross-Attention (Encoder-Decoder Attention)
        cross_attn_output, _ = self.cross_attention(
            x, encoder_output, encoder_output,  # Q: decoder, K,V: encoder
            key_padding_mask=memory_mask        # Encoder의 padding 마스크
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))  
        
        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))  
        
        return x

class Decoder(nn.Module):
    """
    Transformer Decoder
    
    구조:
    [DecoderLayer] * num_layers -> LayerNorm
    
    Args:
        num_layers (int): Decoder Layer의 반복 횟수
        input_size (int): 입력/출력 벡터의 차원 (hidden size)
        num_heads (int): 멀티헤드 어텐션의 헤드 개수
        ff_size (int): FFN 내부 차원
        dropout (float): 드롭아웃 확률
        
    forward Args:
        x (Tensor): [batch_size, target_length, input_size] - Decoder 입력 (이미 임베딩된)
        encoder_output (Tensor): [batch_size, source_length, input_size] - Encoder 출력
        target_mask (Tensor): [target_length, target_length] - 미래 토큰 마스킹
        memory_mask (Tensor): [batch_size, source_length] - Encoder padding 마스크
        
    Returns:
        output (Tensor): [batch_size, target_length, input_size]
    """
    def __init__(self, num_layers, input_size, num_heads, ff_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(input_size, num_heads, ff_size, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, target_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask, memory_mask)
        
        return self.norm(x)
