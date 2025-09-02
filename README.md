# practice_transformer
Transformer 구조를 직접 구현하고 실습하며, 그 동작 원리를 이해하는 것을 목표로 하는 학습용 프로젝트입니다.

본 프로젝트는 `Attention Is All You Need` 논문을 기반으로 하여,
Transformer 모델의 핵심 구조(`Embedding`, `Multi-Head Attention`, `Position-wise FFN`, `Encoder/Decoder` 스택 등)를
직접 코드로 옮기고 실험하는 데 초점을 두고 있습니다.
> 참고: Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.

---

## 구현 초점
- Encoder / Decoder 모듈 (Multi-Head Attention, Feed-Forward Network, Residual + LayerNorm)
- Positional Encoding (sinusoidal)
- Causal Mask (look-ahead mask for autoregressive decoding)
- Padding Mask
- Auto-regressive `generate` 함수 (추론용)

## 코드 구조
```
flowchart TD
    subgraph Encoder["Encoder (stacked N=6)"]
        E1["Multi-Head Self-Attention"] --> EN1["Add & Norm"]
        EN1 --> E2["Feed Forward"] --> EN2["Add & Norm"]
    end
    
    subgraph Decoder["Decoder (stacked N=6)"]
        D1["Masked Multi-Head Self-Attention"] --> DN1["Add & Norm"]
        DN1 --> D2["Cross-Attention (Encoder-Decoder)"] --> DN2["Add & Norm"]
        DN2 --> D3["Feed Forward"] --> DN3["Add & Norm"]
    end
    
    subgraph Model["Transformer"]
        SRC["Source Embedding + Positional Encoding"] --> Encoder
        Encoder --> Decoder
        TGT["Target Embedding + Positional Encoding"] --> Decoder
        Decoder --> PROJ["Linear Projection → Softmax"]
    end
    
    %% 스타일링
    classDef encoderBox fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef decoderBox fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef attentionBox fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef normBox fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
    classDef embeddingBox fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef projectionBox fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#000
    
    %% 클래스 적용
    class E1,D1,D2 attentionBox
    class EN1,EN2,DN1,DN2,DN3 normBox
    class E2,D3 encoderBox
    class SRC,TGT embeddingBox
    class PROJ projectionBox
    
    %% 서브그래프 스타일링
    style Encoder fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style Decoder fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style Model fill:#f5f5f5,stroke:#424242,stroke-width:3px
---