import torch
import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.embedding import TransformerEmbedding

class Transformer(nn.Module):
    """
    완전한 Transformer 모델 (Encoder-Decoder 구조)
    
    Args:
        src_vocab_size (int): 소스 어휘 사전 크기
        tgt_vocab_size (int): 타겟 어휘 사전 크기
        d_model (int): 모델 차원 (기본값: 512)
        num_heads (int): 어텐션 헤드 수 (기본값: 8)
        num_layers (int): 인코더/디코더 레이어 수 (기본값: 6)
        ff_size (int): FFN 내부 차원 (기본값: 2048)
        max_length (int): 최대 시퀀스 길이 (기본값: 5000)
        dropout (float): 드롭아웃 확률 (기본값: 0.1)
        pad_token_id (int): 패딩 토큰 ID (기본값: 0)
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size, 
        d_model=512,
        num_heads=8,
        num_layers=6,
        ff_size=2048,
        max_length=5000,
        dropout=0.2,
        pad_token_id=0
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        # 임베딩 레이어
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_length, dropout)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_length, dropout)
        
        # 인코더 & 디코더 (기존 encoder.py는 dropout 매개변수 없음)
        self.encoder = Encoder(num_layers, d_model, num_heads, ff_size)
        self.decoder = Decoder(num_layers, d_model, num_heads, ff_size, dropout)
        
        # 출력 레이어
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, tokens):
        """패딩 마스크 생성: True = 패딩"""
        return (tokens == self.pad_token_id)
    
    def create_causal_mask(self, size, device):
        """Causal 마스크 생성: True = 마스킹"""
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    
    def forward(self, src_tokens, tgt_tokens):
        """
        Forward pass
        
        Args:
            src_tokens: [batch_size, src_len] - 소스 토큰 인덱스
            tgt_tokens: [batch_size, tgt_len] - 타겟 토큰 인덱스
            
        Returns:
            logits: [batch_size, tgt_len, tgt_vocab_size] - 출력 로짓
        """
        # 마스크 생성
        src_mask = self.create_padding_mask(src_tokens)
        tgt_causal_mask = self.create_causal_mask(tgt_tokens.size(1), tgt_tokens.device)
        memory_mask = src_mask
        
        # 임베딩
        src_embedded = self.src_embedding(src_tokens)
        tgt_embedded = self.tgt_embedding(tgt_tokens)
        
        # 인코더 → 디코더
        encoder_output = self.encoder(src_embedded, src_mask)
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_causal_mask, memory_mask)
        
        # 출력 로짓
        logits = self.output_projection(decoder_output)
        return logits
    
    def generate(self, src_tokens, max_length=50, start_token=1, end_token=2):
        """
        Auto-regressive 생성 (추론용)
        
        Args:
            src_tokens: [batch_size, src_len] - 소스 토큰들
            max_length: 최대 생성 길이
            start_token: 시작 토큰 ID
            end_token: 종료 토큰 ID
            
        Returns:
            generated: [batch_size, generated_len] - 생성된 토큰들
        """
        self.eval()
        with torch.no_grad():
            batch_size = src_tokens.size(0)
            device = src_tokens.device
            
            # 인코더 실행 (한 번만)
            src_mask = self.create_padding_mask(src_tokens)
            src_embedded = self.src_embedding(src_tokens)
            encoder_output = self.encoder(src_embedded, src_mask)
            
            # 시작 토큰으로 초기화
            generated = torch.full((batch_size, 1), start_token, device=device)
            
            for _ in range(max_length - 1):
                # 디코더 실행
                tgt_causal_mask = self.create_causal_mask(generated.size(1), device)
                tgt_embedded = self.tgt_embedding(generated)
                decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_causal_mask, src_mask)
                
                # 다음 토큰 예측
                logits = self.output_projection(decoder_output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # 종료 조건
                if (next_token == end_token).all():
                    break
            
            return generated

def create_model(src_vocab_size, tgt_vocab_size, **kwargs):
    """
    Transformer 모델 생성 헬퍼 함수
    
    Args:
        src_vocab_size: 소스 어휘 사전 크기
        tgt_vocab_size: 타겟 어휘 사전 크기
        **kwargs: 추가 하이퍼파라미터
        
    Returns:
        model: Transformer 모델
    """
    return Transformer(src_vocab_size, tgt_vocab_size, **kwargs)

def count_parameters(model):
    """모델 매개변수 개수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)