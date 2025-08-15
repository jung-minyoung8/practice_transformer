import re
import math
import torch
import torch.nn as nn
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# 텍스트 정제 함수
# 감정 표현(ㅋㅋ, ㅠㅠ, !, ?)은 보존하고 나머지 특수문자 제거
def clean_mt(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^가-힣a-zA-Z0-9\sㅋㅎㅠㅜ!?]', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text):
    cleaned = clean_mt(text)
    if not cleaned:  # 빈 문자열인 경우
        return []
    return cleaned.split()

# 단어 수 세기 (MAX_LEN 추정용)
def count_tokens(example):
    text = clean_mt(example["document"])
    return len(text.split())

# MAX_LEN을 토큰 수의 최댓값으로 설정
def get_max_len(dataset, field='document'):
    lengths = [len(clean_mt(ex[field]).split()) for ex in dataset]
    max_len = max(lengths)
    print(f"최대 길이: {max_len}")
    print(f"여유 공간을 위해 {max_len + 1}을 MAX_LEN으로 설정합니다.")
    return max_len + 1

# Lang 클래스: 단어를 인덱스로 변환하는 사전 생성기 (수정됨)
class Lang:
    def __init__(self, name, min_freq=1):
        self.name = name
        self.min_freq = min_freq
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word2count = {}
        self.n_words = 4

    def add_word(self, tokens):
        for token in tokens:
            if token in self.word2count:
                self.word2count[token] += 1
            else:
                self.word2count[token] = 1

    def build(self):
        """min_freq 이상인 단어들만 vocabulary에 추가"""
        for word, count in self.word2count.items():
            if count >= self.min_freq and word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

# 토큰 인덱싱 + 패딩 처리
def encode_and_pad(tokens, lang, max_len):
    tokens = tokens[:max_len - 1] + ['<EOS>']
    ids = [lang.word2index.get(t, lang.word2index['<UNK>']) for t in tokens]
    ids += [lang.word2index['<PAD>']] * (max_len - len(ids))
    return ids[:max_len]

class TextClsDataset(Dataset):
    def __init__(self, hf_dataset, lang, text_field="document", label_field="label",
                 max_len=128, use_sos=False, use_eos=True):
        self.ds = hf_dataset
        self.lang = lang
        self.text_field = text_field
        self.label_field = label_field
        self.max_len = max_len
        self.use_sos = use_sos
        self.use_eos = use_eos

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        tokens = tokenize(ex[self.text_field])
        
        # 빈 토큰 리스트인 경우 처리
        if not tokens:
            tokens = ['<UNK>']  # 최소한 하나의 토큰은 있어야 함
            
        ids = encode_and_pad(tokens, self.lang, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(ex[self.label_field], dtype=torch.long)

# 수정됨: 메서드 이름과 호출 방식 수정
def build_vocab_from_dataset(hf_dataset, text_field="document", min_freq=1):
    lang = Lang(name="vocab", min_freq=min_freq)
    for ex in hf_dataset:
        tokens = tokenize(ex[text_field])
        if tokens:  # 빈 토큰 리스트가 아닌 경우만 추가
            lang.add_word(tokens)
    lang.build()
    print(f"[vocab] size={lang.n_words} (min_freq={min_freq})")
    return lang

def make_loaders(hf_train, hf_val, text_field="document", label_field="label",
                 min_freq=1, p_len=0.98, extra=1, hard_cap=None,
                 batch_size=32, num_workers=0, use_sos=False, use_eos=True):
    lang = build_vocab_from_dataset(hf_train, text_field=text_field, min_freq=min_freq)
    max_len = get_max_len(hf_train, field=text_field)

    train_ds = TextClsDataset(hf_train, lang, text_field, label_field, max_len, use_sos, use_eos)
    val_ds   = TextClsDataset(hf_val,   lang, text_field, label_field, max_len, use_sos, use_eos)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return lang, max_len, train_loader, val_loader

def load_nsmc_from_dir(data_dir):
    """
    data_dir: ratings_train.txt / ratings_test.txt 가 직접 들어있는 'raw' 폴더 경로
              예) -d ./nsmc/raw
    """
    data_dir = Path(data_dir)

    # 1) 기본: data_dir 안에 파일이 바로 있는지 확인
    train_file = data_dir / "ratings_train.txt"
    test_file  = data_dir / "ratings_test.txt"

    # 2) 백워드 호환: 만약 사용자가 상위 폴더(-d ./ 또는 -d ./nsmc)를 줬다면 자동 보정
    if not (train_file.exists() and test_file.exists()):
        maybe = data_dir / "nsmc" / "raw"
        if (maybe / "ratings_train.txt").exists() and (maybe / "ratings_test.txt").exists():
            train_file = maybe / "ratings_train.txt"
            test_file  = maybe / "ratings_test.txt"
        else:
            # raw 폴더가 직접 있는지 확인
            maybe_raw = data_dir / "raw"
            if (maybe_raw / "ratings_train.txt").exists() and (maybe_raw / "ratings_test.txt").exists():
                train_file = maybe_raw / "ratings_train.txt"
                test_file  = maybe_raw / "ratings_test.txt"

    # 3) 최종 확인
    if not (train_file.exists() and test_file.exists()):
        raise FileNotFoundError(
            "NSMC 파일을 찾을 수 없습니다. -d 로 올바른 폴더를 지정하세요.\n"
            f"시도한 경로:\n- {train_file}\n- {test_file}"
        )

    ds = load_dataset(
        "csv",
        data_files={"train": str(train_file), "test": str(test_file)},
        delimiter="\t",
    )
    if "id" in ds["train"].column_names:
        ds = ds.remove_columns(["id"])
    return ds["train"], ds["test"]

def make_loaders_from_dir(data_dir, **kwargs):
    """
    data_dir: ratings_train.txt / ratings_test.txt 가 들어있는 폴더
              (예: ./nsmc/raw)
    kwargs: make_loaders 인자들과 동일
    """
    train, test = load_nsmc_from_dir(data_dir)
    return make_loaders(train, test, **kwargs)