import torch
import torch.nn as nn
import math

def get_positional_encoding(seq_len,
                        input_size,
                        min_timescale=1.0,
                        max_timescale=1.0e4,
                        start_index=0,
                        device='cpu'):
    position = torch.arange(seq_len, dtype=torch.float, device=device)
    num_timescales = input_size // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(1, num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float, device=device) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)  # 명시
    pos_en = torch.zeros(seq_len, input_size, device=device)
    pos_en[:, 0::2] = torch.sin(scaled_time)
    pos_en[:, 1::2] = torch.cos(scaled_time)
    return pos_en

class TransformerEmbedding(nn.Module):
    """
    Token Embedding + Positional Encoding을 결합한 모듈
    
    Args:
        vocab_size (int): 어휘 사전 크기
        d_model (int): 모델 차원
        max_length (int): 최대 시퀀스 길이
        dropout (float): 드롭아웃 확률

    Returns:
        Tensor: [batch_size, seq_len, d_model]
    """
    def __init__(self, vocab_size, d_model, max_length=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Token Embedding & positional encoding
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        pos_enc = get_positional_encoding(
            seq_len=seq_len,
            input_size=self.d_model,
            device=x.device
        )
        
        # 임베딩 + 위치 인코딩
        embedded = embedded + pos_enc.unsqueeze(0)
        
        return self.dropout(embedded)