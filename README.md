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
```mermaid
flowchart TD
    %% Nodes outside subgraphs
    SRC["Source Embedding + Positional Encoding"]
    TGT["Target Embedding + Positional Encoding"]
    PROJ["Linear Projection -> Softmax"]

    %% Encoder
    subgraph ENC["Encoder (stacked N=6)"]
        E1["Multi-Head Self-Attention"] --> EN1["Add & Norm"]
        EN1 --> E2["Feed Forward"] --> EN2["Add & Norm"]
    end

    %% Decoder
    subgraph DEC["Decoder (stacked N=6)"]
        D1["Masked Multi-Head Self-Attention"] --> DN1["Add & Norm"]
        DN1 --> D2["Cross-Attention (Encoder-Decoder)"] --> DN2["Add & Norm"]
        DN2 --> D3["Feed Forward"] --> DN3["Add & Norm"]
    end

    %% Model-level wiring
    SRC --> ENC
    ENC --> DEC
    TGT --> DEC
    DEC --> PROJ

    %% Styling
    style E1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style D1 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style D2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style EN1 fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style EN2 fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style DN1 fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style DN2 fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style DN3 fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style E2 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style D3 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style SRC fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style TGT fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style PROJ fill:#fff8e1,stroke:#ffa000,stroke-width:2px
```

---