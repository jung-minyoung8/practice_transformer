# practice_transformer
Transformer 구조를 직접 구현하고 실습하며, 그 동작 원리를 이해하는 것을 목표로 하는 학습용 프로젝트입니다.

본 프로젝트는 `Attention Is All You Need` 논문을 기반으로 하여,
Transformer 모델의 핵심 구조(`Embedding`, `Multi-Head Attention`, `Position-wise FFN`, `Encoder/Decoder` 스택 등)를
직접 코드로 옮기고 실험하는 데 초점을 두고 있습니다.
> 참고: Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017.

---

## 환경 구성 파이프라인
Python 버전
- 권장 버전: `Python` 3.10 이상
- 가상환경 사용을 권장합니다.
### 1. Conda 가상환경 생성 및 활성화
```
conda create -n transformer python=3.10 -y
conda activate transformer
python -m pip install --upgrade pip
```
### 2. requirements.txt 기반 설치
```
pip install -r requirements.txt
```
### 3. PyTorch 설치 (CUDA 12.1 버전)
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
---