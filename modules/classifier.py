import torch
import torch.nn as nn
from modules import Encoder, get_positional_encoding


class EncoderOnlyBackbone(nn.Module):
    """
    Encoder-Only Transformer Backbone
    
    구조:
    [Token Embedding + Positional Encoding] -> [EncoderLayer] * num_layers -> LayerNorm
    
    - 토큰 임베딩 + 사인/코사인 포지셔널 인코딩 (+드롭아웃)
    - bidirectional self-attention이 있는 인코더 블록 스택
    - 출력은 마지막 레이어의 히든 시퀀스
    
    Args:
        vocab_size (int): 어휘 크기
        d_model (int): 입력/출력 벡터의 차원 (hidden size)
        num_heads (int): Multi-Head Attention의 head 개수
        ff_size (int): FFN 내부 차원
        num_layers (int): Encoder Layer의 반복 횟수
        pad_id (int): 패딩 토큰 ID
        dropout (float): 드롭아웃 비율
        max_len (int): 최대 시퀀스 길이
        
    forward Args:
        input_ids (Tensor): [batch_size, seq_length] (패딩은 보통 오른쪽, [CLS] 토큰 추가 가능)
        
    Returns:
        output (Tensor): [batch_size, seq_length, d_model]
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, ff_size: int,
                 num_layers: int, pad_id: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 기존 Encoder 사용
        self.encoder = Encoder(num_layers, d_model, num_heads, ff_size)
        self.max_len = max_len
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_length]
        Returns:
            output: [batch_size, seq_length, d_model]
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Padding mask 생성 (True=pad)
        padding_mask = (input_ids == self.pad_id)
        
        # Token embedding (√d_model 스케일링)
        x = self.emb(input_ids) * (self.d_model ** 0.5)  # [B, L, D]
        
        # Positional encoding
        pos = get_positional_encoding(
            seq_len=L, input_size=self.d_model, device=device
        )
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)  # [1, L, D]
        x = x + pos[:, :L, :].to(x.dtype)
        x = self.dropout(x)
        
        # Encoder forward
        x = self.encoder(x, mask=padding_mask)  # [B, L, D]
        
        return x


class EncoderOnlyClassifier(nn.Module):
    """
    Encoder-Only Transformer Classifier (BERT-style)
    
    구조:
    [EncoderOnlyBackbone] -> [CLS Token Hidden 또는 Pooling] -> [Dropout + Linear]
    
    - [CLS] 토큰의 히든 또는 평균 풀링을 사용하여 분류
    
    Args:
        vocab_size (int): 어휘 크기
        d_model (int): 입력/출력 벡터의 차원 (hidden size)
        num_heads (int): Multi-Head Attention의 head 개수
        ff_size (int): FFN 내부 차원
        num_layers (int): Encoder Layer의 반복 횟수
        pad_id (int): 패딩 토큰 ID
        num_labels (int): 분류할 클래스 개수
        dropout (float): 드롭아웃 비율
        pooling_strategy (str): 'cls' 또는 'mean' - 풀링 전략
        
    forward Args:
        input_ids (Tensor): [batch_size, seq_length] 
                          - pooling_strategy='cls': 맨 앞에 [CLS] 토큰 위치
                          - pooling_strategy='mean': 패딩 제외하고 평균
        
    Returns:
        logits (Tensor): [batch_size, num_labels]
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, ff_size: int,
                 num_layers: int, pad_id: int, num_labels: int, 
                 dropout: float = 0.1, pooling_strategy: str = 'cls'):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.pad_id = pad_id
        
        self.backbone = EncoderOnlyBackbone(
            vocab_size, d_model, num_heads, ff_size, num_layers, pad_id, dropout
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels)
        )
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_length]
        Returns:
            logits: [batch_size, num_labels]
        """
        # Backbone forward
        x = self.backbone(input_ids)   # [B, L, D]
        
        if self.pooling_strategy == 'cls':
            # [CLS] 토큰 히든 사용 (첫 번째 토큰)
            pooled = x[:, 0, :]  # [B, D]
        elif self.pooling_strategy == 'mean':
            # 패딩 제외하고 평균 풀링
            padding_mask = (input_ids == self.pad_id)  # [B, L]
            lengths = (~padding_mask).sum(dim=1, keepdim=True).float()  # [B, 1]
            masked_x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)  # [B, L, D]
            pooled = masked_x.sum(dim=1) / lengths  # [B, D]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # 분류
        logits = self.classifier(pooled)  # [B, num_labels]
        return logits