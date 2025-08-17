import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from pathlib import Path

from config import config
from transformer import create_model, count_parameters
from data_utils import make_translation_loaders_from_dir

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

def train_epoch(model, train_loader, optimizer, criterion, device):
    """한 에포크 훈련"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        src_ids = batch['src_ids'].to(device)
        tgt_ids = batch['tgt_ids'].to(device)
        
        # Teacher forcing: 입력과 타겟 분리
        tgt_input = tgt_ids[:, :-1]  # <sos> ~ 마지막 이전까지
        tgt_output = tgt_ids[:, 1:]  # 첫 번째 다음 ~ <eos>까지
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input)
        
        # Loss 계산 (패딩 토큰 제외)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
        optimizer.step()
        
        # 통계 업데이트
        total_loss += loss.item()
        
        # Progress bar 업데이트
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
        
        # 주기적 로그 출력
        if config.VERBOSE and (batch_idx + 1) % config.LOG_INTERVAL == 0:
            print(f"    Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

def evaluate(model, val_loader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
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
    
    return total_loss / num_batches

def translate_sample(model, src_text, src_lang, tgt_lang, device):
    """샘플 번역 테스트"""
    from data_utils import tokenize, encode_and_pad
    
    model.eval()
    
    try:
        with torch.no_grad():
            # 소스 문장 토크나이징 및 인코딩
            src_tokens = tokenize(src_text)
            if not src_tokens:
                src_tokens = ['<UNK>']
            
            # 적당한 길이로 인코딩 (128은 일반적인 최대 길이)
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

def run_translation_tests(model, src_lang, tgt_lang, device):
    """번역 테스트 실행"""
    print("\n🔍 번역 테스트:")
    print("-" * 40)
    
    for sentence in config.TEST_SENTENCES:
        translated = translate_sample(model, sentence, src_lang, tgt_lang, device)
        print(f"  {config.SRC_FIELD}: {sentence}")
        print(f"  {config.TGT_FIELD}: {translated}")
        print()

def save_model(model, optimizer, epoch, train_loss, val_loss, src_lang, tgt_lang, filepath):
    """모델 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'config': config
    }, filepath)

def plot_training_progress(train_losses, val_losses, save_path):
    """훈련 진행 상황 그래프 생성"""
    if not config.SAVE_PLOTS:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 최근 에포크 확대 그래프
    plt.subplot(1, 2, 2)
    start_epoch = max(0, len(train_losses) - 5)  # 최근 5 에포크
    epochs = range(start_epoch, len(train_losses))
    plt.plot(epochs, train_losses[start_epoch:], label='Train Loss', color='blue', alpha=0.7, marker='o')
    plt.plot(epochs, val_losses[start_epoch:], label='Validation Loss', color='red', alpha=0.7, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Recent Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # 데이터 로더 생성
    print("\n📚 데이터 로딩 중...")
    try:
        src_lang, tgt_lang, src_max_len, tgt_max_len, train_loader, val_loader = \
            make_translation_loaders_from_dir(
                config.DATA_DIR,
                train_filename=config.TRAIN_FILENAME,
                val_filename=config.VALID_FILENAME,
                src_field=config.SRC_FIELD,
                tgt_field=config.TGT_FIELD,
                min_freq=config.MIN_FREQ,
                batch_size=config.BATCH_SIZE,
                num_workers=config.NUM_WORKERS
            )
    except FileNotFoundError as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        print("\n📋 필요한 파일:")
        print(f"- {config.DATA_DIR}/{config.TRAIN_FILENAME}.json")
        print(f"- {config.DATA_DIR}/{config.VALID_FILENAME}.json")
        return
    
    print(f"✅ 데이터 로딩 완료!")
    print(f"   소스 어휘 사전 크기: {src_lang.n_words:,}")
    print(f"   타겟 어휘 사전 크기: {tgt_lang.n_words:,}")
    print(f"   훈련 배치 수: {len(train_loader):,}")
    print(f"   검증 배치 수: {len(val_loader):,}")
    
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
    
    total_params = count_parameters(model)
    print(f"✅ 모델 생성 완료!")
    print(f"   총 매개변수 수: {total_params:,}")
    print(f"   모델 크기: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Loss function & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_lang.word2index['<PAD>'])
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=config.BETAS,
        eps=config.EPS
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=config.SCHEDULER_PATIENCE, 
        factor=config.SCHEDULER_FACTOR,
        verbose=True
    )
    
    # 훈련 루프
    print(f"\n🔥 훈련 시작! ({config.NUM_EPOCHS} 에포크)")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_msg = f"📊 Epoch {epoch + 1}/{config.NUM_EPOCHS}"
        print(f"\n{epoch_start_msg}")
        print("-" * len(epoch_start_msg))
        
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        train_losses.append(train_loss)
        
        # 평가
        if (epoch + 1) % config.EVAL_EVERY == 0:
            val_loss = evaluate(model, val_loader, criterion, config.DEVICE)
            val_losses.append(val_loss)
        else:
            val_loss = val_losses[-1] if val_losses else float('inf')
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 결과 출력
        print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        # 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = save_dir / "best_model.pt"
            save_model(model, optimizer, epoch, train_loss, val_loss, src_lang, tgt_lang, best_model_path)
            print(f"💾 최고 성능 모델 저장! (Val Loss: {val_loss:.4f})")
        
        # 주기적 모델 저장
        if (epoch + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = save_dir / f"model_epoch_{epoch + 1}.pt"
            save_model(model, optimizer, epoch, train_loss, val_loss, src_lang, tgt_lang, checkpoint_path)
        
        # 번역 테스트
        if (epoch + 1) % config.TRANSLATE_EVERY == 0:
            run_translation_tests(model, src_lang, tgt_lang, config.DEVICE)
        
        # 훈련 그래프 업데이트
        if config.SAVE_PLOTS and len(val_losses) > 0:
            plot_path = save_dir / "training_progress.png"
            plot_training_progress(train_losses, val_losses, plot_path)
    
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
    if config.SAVE_PLOTS:
        final_plot_path = save_dir / "final_training_progress.png"
        plot_training_progress(train_losses, val_losses, final_plot_path)
        print(f"📊 훈련 그래프 저장: {final_plot_path}")
    
    print(f"\n✅ 모든 작업 완료! 🎊")

if __name__ == "__main__":
    main()