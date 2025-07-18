import re
import math
import torch
import torch.nn as nn
from datasets import load_dataset

# 텍스트 정제 함수
# 감정 표현(ㅋㅋ, ㅠㅠ, !, ?)은 보존하고 나머지 특수문자 제거
def clean_mt(text):
    text = text.lower()
    text = re.sub(r'[^가-힣a-zA-Z0-9\sㅋㅎㅠㅜ!?]', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 단어 수 세기 (MAX_LEN 추정용)
def count_tokens(example):
    text = clean_mt(example["document"])
    return len(text.split())
# MAX_LEN을 토큰 수의 최댓값으로 설정
def get_max_len(dataset, field='documet'):
    lengths = [len(clean_mt(ex[field]).split()) for ex in dataset]
    max_len = max(lengths)
    print(f"최대 길이: {max_len}")
    print(f"여유 공간을 위해 {max_len + 1}을 MAX_LEN으로 설정합니다.")
    return max_len + 1

# Lang 클래스: 단어를 인덱스로 변환하는 사전 생성기
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.index2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
        self.word2count = {}
        self.n_words = 4

    def add_word(self, tokens):
        for token in tokens:
            if token not in self.word2index:
                self.word2index[token] = self.n_words
                self.index2word[self.n_words] = token
                self.word2count[token] = 1
                self.n_words += 1
            else:
                self.word2count[token] += 1

# 토큰 인덱싱 + 패딩 처리
def encode_and_pad(tokens, lang, max_len):
    tokens = tokens[:max_len - 1] + ['<EOS>']
    ids = [lang.word2index.get(t, lang.word2index['<UNK>']) for t in tokens]
    ids += [lang.word2index['<PAD>']] * (max_len - len(ids))
    return ids[:max_len]

