import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import json
from pathlib import Path
from datasets import Dataset
from torch.utils.data import DataLoader

from config import config
from transformer import create_model, count_parameters
from data_utils import Lang, tokenize, encode_and_pad

def set_seed(seed):
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_translation_data(data_dir, train_filename, val_filename):
    """번역 데이터 로드 (main.py 내부용)"""
    data_dir = Path(data_dir)
    
    train_file = data_dir / f"{train_filename}.json"
    val_file = data_dir / f"{val_filename}.json"
    
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            f"필요한 파일:\n- {train_file}\n- {val_file}"
        )
    
    print(f"훈련 데이터: {train_file}")
    print(f"검증 데이터: {val_file}")
    
    # JSON 파일 로드
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # data 필드 추출
    if isinstance(train_data, dict) and 'data' in train_data:
        train_data = train_data['data']
    if isinstance(val_data, dict) and 'data' in val_data:
        val_data = val_data['data']
    
    # 딕셔너리 형태로 변환
    train_dict = {
        'korean': [item['ko'] for item in train_data],
        'english': [item['en'] for item in train_data]
    }
    val_dict = {
        'korean': [item['ko'] for item in val_data],
        'english': [item['en'] for item in val_data]
    }
    
    # datasets.Dataset으로 변환
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    
    return train_dataset, val_dataset

def build_vocab(dataset, field, min_freq):
    """어휘 사전 구축"""
    lang = Lang(name=field, min_freq=min_freq)
    for ex in dataset:
        tokens = tokenize(ex[field])
        if tokens:
            lang.add_word(tokens)
    lang.build()
    return lang

def get_max_length(dataset, field):
    """최대 길이 계산"""
    lengths = [len(tokenize(ex[field])) for ex in dataset]
    return max(lengths) + 2  # SOS, EOS 고려

class TranslationDataset:
    """간단한 번역 데이터셋"""
    def __init__(self, dataset, src_lang, tgt_lang, src_max_len, tgt_max_len):
        self.dataset = dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        ex = self.dataset[i]
        
        # 소스 처리
        src_tokens = tokenize(ex['korean'])
        if not src_tokens:
            src_tokens = ['<UNK>']
        src_ids = encode_and_pad(src_tokens, self.src_lang, self.src_max_len)
        
        # 타겟 처리
        tgt_tokens = tokenize(ex['english'])
        if not tgt_tokens:
            tgt_tokens = ['<UNK>']
        
        # <SOS> + tokens + <EOS> 형태
        tgt_tokens = ['<SOS>'] + tgt_tokens[:self.tgt_max_len - 2] + ['<EOS>']
        tgt_ids = [self.tgt_lang.word2index.get(t, self.tgt_lang.word2index['<UNK>']) for t in tgt_tokens]
        tgt_ids += [self.tgt_lang.word2index['<PAD>']] * (self.tgt_max_len - len(tgt_ids))
        tgt_ids = tgt_ids[:self.tgt_max_len]
        
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long)
        }

def train_epoch(model, train_loader, optimizer, criterion, device):
    """한 에포크 훈련"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        src_ids = batch['src_ids'].to(device)
        tgt_ids = batch['tgt_ids'].to(device)
        
        # Teacher forcing
        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input)
        
        # Loss 계산
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    return total_loss / num_batches

def evaluate(model, val_loader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
        
        for batch in progress_bar:
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            
            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]
            
            logits = model(src_ids, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(val_loader)

def translate_sample(model, src_text, src_lang, tgt_lang, device):
    """샘플 번역 테스트"""
    model.eval()
    
    try:
        with torch.no_grad():
            # 소스 문장 처리
            src_tokens = tokenize(src_text)
            if not src_tokens:
                src_tokens = ['<UNK>']
            
            src_ids = encode_and_pad(src_tokens, src_lang, 128)
            src_tensor = torch.tensor([src_ids], device=device)
            
            # 생성
            generated = model.generate(
                src_tensor, 
                max_length=config.GENERATION_MAX_LENGTH,
                start_token=tgt_lang.word2index['<SOS>'], 
                end_token=tgt_lang.word2index['<EOS>']
            )
            
            # 디코딩
            generated_ids = generated[0].cpu().tolist()
            translated_tokens = []
            for id in generated_ids:
                if id == tgt_lang.word2index['<EOS>']:
                    break
                if id not in [tgt_lang.word2index['<PAD>'], tgt_lang.word2index['<SOS>']]:
                    word = tgt_lang.index2word.get(id, '<UNK>')
                    translated_tokens.append(word)
            
            return ' '.join(translated_tokens)
    
    except Exception as e:
        return f"[번역 실패: {str(e)}]"

def plot_training_progress(train_losses, val_losses, save_path=None):
    """훈련 진행 상황 그래프"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"📊 그래프 저장: {save_path}")
    
    plt.show()

def run_translation_tests(model, src_lang, tgt_lang, device):
    """번역 테스트 실행"""
    test_sentences = [
        "안녕하세요.",
        "오늘 날씨가 좋습니다.",
        "저는 학생입니다.",
        "감사합니다.",
        "좋은 하루 보내세요."
    ]
    
    for sentence in test_sentences:
        translated = translate_sample(model, sentence, src_lang, tgt_lang, device)
        print(f"한국어: {sentence}")
        print(f"영어: {translated}")
        print("-" * 40)

def main():
    """메인 훈련 함수"""
    
    # 설정 출력
    config.print_config()
    
    # 시드 설정
    set_seed(config.RANDOM_SEED)
    
    # 저장 디렉토리 생성
    save_dir = Path(config.SAVE_DIR)
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n🚀 Transformer 번역 모델 훈련 시작!")
    print(f"디바이스: {config.DEVICE}")
    
    # 데이터 로드
    print("\n📚 데이터 로딩 중...")
    try:
        train_dataset, val_dataset = load_translation_data(
            config.DATA_DIR, config.TRAIN_FILENAME, config.VALID_FILENAME
        )
        
        print(f"훈련 데이터 크기: {len(train_dataset):,}")
        print(f"검증 데이터 크기: {len(val_dataset):,}")
        
        # 어휘 사전 구축
        print("어휘 사전 구축 중...")
        src_lang = build_vocab(train_dataset, 'korean', config.MIN_FREQ)
        tgt_lang = build_vocab(train_dataset, 'english', config.MIN_FREQ)
        
        print(f"소스 어휘 사전 크기: {src_lang.n_words:,}")
        print(f"타겟 어휘 사전 크기: {tgt_lang.n_words:,}")
        
        # 최대 길이 계산
        src_max_len = get_max_length(train_dataset, 'korean')
        tgt_max_len = get_max_length(train_dataset, 'english')
        
        # 데이터셋 및 로더 생성
        train_ds = TranslationDataset(train_dataset, src_lang, tgt_lang, src_max_len, tgt_max_len)
        val_ds = TranslationDataset(val_dataset, src_lang, tgt_lang, src_max_len, tgt_max_len)
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
    except FileNotFoundError as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return
    
    # 모델 생성
    print("\n🏗️ 모델 생성 중...")
    model = create_model(
        src_vocab_size=src_lang.n_words,
        tgt_vocab_size=tgt_lang.n_words,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        ff_size=config.FF_SIZE,
        dropout=config.DROPOUT,
        pad_token_id=src_lang.word2index['<PAD>']
    )
    
    model = model.to(config.DEVICE)
    print(f"모델 매개변수 수: {count_parameters(model):,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_lang.word2index['<PAD>'])
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # 훈련 루프
    print(f"\n🔥 훈련 시작! ({config.NUM_EPOCHS} 에포크)")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n📊 Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        train_losses.append(train_loss)
        
        # 평가
        val_loss = evaluate(model, val_loader, criterion, config.DEVICE)
        val_losses.append(val_loss)
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'config': config
            }, save_dir / 'best_model.pt')
            print("💾 최고 성능 모델 저장!")
        
        # 번역 테스트
        if (epoch + 1) % config.TRANSLATE_EVERY == 0:
            print("\n🔍 번역 테스트:")
            for sentence in config.TEST_SENTENCES[:3]:  # 처음 3개만
                translated = translate_sample(model, sentence, src_lang, tgt_lang, config.DEVICE)
                print(f"  한국어: {sentence}")
                print(f"  영어: {translated}")
                print()
    
    # 훈련 완료
    print("\n" + "=" * 60)
    print("🎉 훈련 완료!")
    print(f"📈 최고 검증 손실: {best_val_loss:.4f}")
    print(f"💾 모델 저장 위치: {save_dir}")
    
    # 최종 번역 테스트
    print("\n🔍 최종 번역 테스트:")
    print("=" * 60)
    run_translation_tests(model, src_lang, tgt_lang, config.DEVICE)
    
    # 최종 그래프 저장
    plot_path = save_dir / "training_progress.png"
    plot_training_progress(train_losses, val_losses, plot_path)
    
    print(f"\n✅ 모든 작업 완료! 🎊")

if __name__ == "__main__":
    main()