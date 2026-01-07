"""
Script de treinamento da CNN
"""

import os
import time
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Desabilitar verificação SSL (necessário para redes corporativas)
ssl._create_default_https_context = ssl._create_unverified_context

from model import create_model
from utils import (
    get_device,
    get_transforms,
    save_checkpoint,
    AverageMeter,
    accuracy,
    plot_training_history
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Treina o modelo por uma época
    
    Args:
        model: Modelo CNN
        train_loader: DataLoader de treinamento
        criterion: Função de loss
        optimizer: Otimizador
        device: Device (CPU/GPU)
        epoch: Número da época atual
        
    Returns:
        loss médio, acurácia média
    """
    model.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Época {epoch} [TREINO]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Métricas
        acc = accuracy(outputs, labels)[0]
        losses.update(loss.item(), images.size(0))
        accs.update(acc.item(), images.size(0))
        
        # Atualizar barra de progresso
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accs.avg:.2f}%'
        })
    
    return losses.avg, accs.avg


def validate(model, val_loader, criterion, device, epoch):
    """
    Valida o modelo
    
    Args:
        model: Modelo CNN
        val_loader: DataLoader de validação
        criterion: Função de loss
        device: Device (CPU/GPU)
        epoch: Número da época atual
        
    Returns:
        loss médio, acurácia média
    """
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    pbar = tqdm(val_loader, desc=f'Época {epoch} [VALID]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Métricas
            acc = accuracy(outputs, labels)[0]
            losses.update(loss.item(), images.size(0))
            accs.update(acc.item(), images.size(0))
            
            # Atualizar barra de progresso
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{accs.avg:.2f}%'
            })
    
    return losses.avg, accs.avg


def train_model(
    num_epochs=50,
    batch_size=128,
    learning_rate=0.001,
    data_dir='./data',
    save_dir='./models',
    resume=False
):
    """
    Função principal de treinamento
    
    Args:
        num_epochs: Número de épocas
        batch_size: Tamanho do batch
        learning_rate: Taxa de aprendizado
        data_dir: Diretório dos dados
        save_dir: Diretório para salvar modelos
        resume: Se True, continua de checkpoint
    """
    # Criar diretórios
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = get_device()
    
    # Datasets e DataLoaders
    print("\n📦 Carregando dataset CIFAR-10...")
    
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✓ Dataset carregado: {len(train_dataset)} treino, {len(val_dataset)} validação")
    
    # Modelo
    print("\n🧠 Criando modelo...")
    model = create_model(num_classes=10)
    model = model.to(device)
    print(f"✓ Modelo com {model.count_parameters():,} parâmetros")
    
    # Loss e Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Histórico
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    start_epoch = 1
    
    # Treinamento
    print(f"\n🚀 Iniciando treinamento por {num_epochs} épocas...\n")
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        
        # Treinar
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validar
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print resumo
        print(f"\n{'='*70}")
        print(f"Época {epoch}/{num_epochs} - Tempo: {epoch_time:.2f}s")
        print(f"Treino    → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Validação → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print(f"{'='*70}\n")
        
        # Salvar melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(save_dir, 'cnn_cifar10_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Novo melhor modelo salvo! Acurácia: {best_acc:.2f}%\n")
        
        # Salvar checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    # Salvar modelo final
    final_model_path = os.path.join(save_dir, 'cnn_cifar10_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Modelo final salvo em: {final_model_path}")
    
    # Plotar histórico
    print("\n📊 Gerando gráficos de treinamento...")
    fig = plot_training_history(history)
    fig.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Gráficos salvos em: {save_dir}/training_history.png")
    
    print(f"\n🎉 Treinamento concluído!")
    print(f"Melhor acurácia de validação: {best_acc:.2f}%")
    
    return model, history


if __name__ == "__main__":
    # Hiperparâmetros
    config = {
        'num_epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.001,
        'data_dir': './data',
        'save_dir': './models'
    }
    
    print("="*70)
    print("CNN PARA CLASSIFICAÇÃO DE IMAGENS - CIFAR-10")
    print("="*70)
    print("\nConfigurações:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Treinar
    model, history = train_model(**config)
