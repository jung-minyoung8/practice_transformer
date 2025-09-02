# practice_transformer
Transformer 구조를 직접 구현하고 실습하며, 그 동작 원리를 이해하는 것을 목표로 하는 학습용 프로젝트입니다.

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