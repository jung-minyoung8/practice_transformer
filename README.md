# practice_transformer

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
## Encoder Only (Classifier)

NSMC 기반 텍스트 분류 프로젝트

이 프로젝트는 `네이버 영화 리뷰 감성 분석(NSMC, Naver Sentiment Movie Corpus v1.0)`을 활용하여,<br>
Transformer Encoder-Only 구조(Classifier) 와 LSTM 모델을 각각 학습 및 비교하는 실험을 수행합니다.

### 데이터셋

이 프로젝트에서는 `Naver sentiment movie corpus v1.0`를 사용했습니다.<br>
Reviews were scraped from [Naver Movies](http://movie.naver.com/movie/point/af/list.nhn).<br>
The dataset construction is based on the method noted in [Large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) from Maas et al., 2011.

#### 데이터 설명
- 파일 형식: 탭 구분 텍스트 (.txt, 실제 구조는 .tsv)
- 컬럼 구성:
  - `id`: 리뷰 ID
  - `document`: 리뷰 내용 (한국어)
  - `label`: 감성 레이블 (0 = 부정, 1 = 긍정)
- 데이터 규모:
  - `ratings.txt`: 전체 200,000개 리뷰
  - `ratings_train.txt`: 학습용 150,000개 리뷰
  - `ratings_test.txt`: 테스트용 50,000개 리뷰

### 실행 방법
1) Encoder-Only 모델 (Transformer Classifier)
```
python encoder_train.py -d ./nsmc
```
`-d`: directory

2) LSTM 모델
```
python lstm_train.py -d ./nsmc
```
`-d`: directory


