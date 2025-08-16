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

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, 
                 num_classes=2, dropout=0.3):
        super(SentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # bidirectional이므로 *2
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 마지막 hidden state 사용 (양방향이므로 forward + backward)
        # hidden: (num_layers*2, batch_size, hidden_dim)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch_size, hidden_dim*2)
        
        # Dropout + Classification
        output = self.dropout(last_hidden)
        logits = self.classifier(output)  # (batch_size, num_classes)
        
        return logits

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
    parser = argparse.ArgumentParser(description='NSMC lstm Analysis Training')
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                       help='ratings_train.txt와 ratings_test.txt가 있는 디렉토리 경로')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=10, help='에포크 수')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--embed_dim', type=int, default=128, help='임베딩 차원')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM 히든 차원')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM 레이어 수')
    parser.add_argument('--dropout', type=float, default=0.3, help='드롭아웃 비율')
    parser.add_argument('--min_freq', type=int, default=2, help='단어 최소 빈도')
    parser.add_argument('--save_path', type=str, default='./models/lstm_model.pth',
                       help='모델 저장 경로')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto, cpu, cuda)')
    
    args = parser.parse_args()
    print(f"파싱된 인자들: {args}")
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"사용 디바이스: {device}")
    
    # 데이터 로더 생성
    print("데이터 로딩 중...")
    try:
        lang, max_len, train_loader, val_loader = make_loaders_from_dir(
            args.data_dir,
            batch_size=args.batch_size,
            min_freq=args.min_freq,
            text_field="document",
            label_field="label"
        )
    except FileNotFoundError as e:
        print(f"오류: {e}")
        return
    
    print(f"어휘 크기: {lang.n_words}")
    print(f"최대 시퀀스 길이: {max_len}")
    print(f"훈련 배치 수: {len(train_loader)}")
    print(f"검증 배치 수: {len(val_loader)}")
    
    # 모델 생성
    model = SentimentClassifier(
        vocab_size=lang.n_words,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # 손실 함수와 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련 시작
    best_val_acc = 0
    print(f"\n훈련 시작 (총 {args.epochs} 에포크)")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 훈련
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, lang, max_len, args.save_path, epoch+1, train_acc, val_acc)
            print(f"새로운 최고 성능! (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nlstm 훈련 완료! 최고 검증 정확도: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()