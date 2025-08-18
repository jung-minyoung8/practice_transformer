import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path

from data_utils import make_loaders_from_dir
from modules.classifier import EncoderOnlyClassifier

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def save_model(model, lang, max_len, save_path, epoch, train_acc, val_acc):
    """모델과 메타데이터를 저장"""
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 저장
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'vocab_size': lang.n_words,
        'max_len': max_len,
    }, save_path)
    
    # vocab 저장
    vocab_path = save_path.replace('.pth', '_vocab.json')
    vocab_data = {
        'word2index': lang.word2index,
        'index2word': lang.index2word,
        'n_words': lang.n_words,
        'max_len': max_len
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"모델 저장: {save_path}")
    print(f"어휘 저장: {vocab_path}")

def main():
    parser = argparse.ArgumentParser(description='Original Transformer Training')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='데이터 디렉토리')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=10, help='에포크 수')
    parser.add_argument('--lr', type=float, default=0.0001, help='학습률')
    parser.add_argument('--d_model', type=int, default=64, help='모델 차원')
    parser.add_argument('--num_heads', type=int, default=8, help='어텐션 헤드 수')
    parser.add_argument('--ff_size', type=int, default=256, help='FFN 차원')
    parser.add_argument('--num_layers', type=int, default=2, help='레이어 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    parser.add_argument('--min_freq', type=int, default=2, help='단어 최소 빈도')
    parser.add_argument('--save_path', type=str, default='./models/original_transformer_model.pth', help='모델 저장 경로')
    parser.add_argument('--device', type=str, default='auto', help='디바이스')
    parser.add_argument('--pooling_strategy', type=str, default='mean', help='풀링 전략')
    
    args = parser.parse_args()
    print(f"파싱된 인자들: {args}")
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"사용 디바이스: {device}")
    
    print("데이터 로딩 중...")
    try:
        lang, max_len, train_loader, val_loader = make_loaders_from_dir(
            args.data_dir, batch_size=args.batch_size, min_freq=args.min_freq,
            text_field="document", label_field="label"
        )
    except Exception as e:
        print(f"오류: {e}")
        return
    
    print(f"어휘 크기: {lang.n_words}")
    print(f"최대 시퀀스 길이: {max_len}")
    print(f"훈련 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")
    
    print("DEBUG: 원본 Transformer 모델 생성 중...")
    model = EncoderOnlyClassifier(
        vocab_size=lang.n_words,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        num_layers=args.num_layers,
        pad_id=0,
        num_labels=2,
        dropout=args.dropout,
        pooling_strategy=args.pooling_strategy
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_acc = 0
    print(f"\nTransformer 훈련 시작 (총 {args.epochs} 에포크)")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, lang, max_len, args.save_path, epoch+1, train_acc, val_acc)
            print(f"새로운 최고 성능! (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTransformer 훈련 완료! 최고 검증 정확도: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()