"""
Funções utilitárias para o projeto
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


# Classes do CIFAR-10
CIFAR10_CLASSES = [
    'avião', 'automóvel', 'pássaro', 'gato', 'cervo',
    'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão'
]

# Classes em inglês (originais)
CIFAR10_CLASSES_EN = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_device():
    """
    Detecta e retorna o device disponível (GPU ou CPU)
    
    Returns:
        torch.device: CUDA se disponível, caso contrário CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU disponível: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ GPU não disponível. Usando CPU.")
    
    return device


def get_transforms(train=True, augmentation_level='basic'):
    """
    Retorna transformações para pré-processar imagens
    
    Args:
        train: Se True, aplica data augmentation
        augmentation_level: 'basic', 'medium', 'strong'
        
    Returns:
        torchvision.transforms.Compose
    """
    if train:
        if augmentation_level == 'basic':
            # Data augmentation básica (original)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ])
        
        elif augmentation_level == 'medium':
            # Augmentation média - RECOMENDADO para avião/caminhão
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(15),  # Rotação ajuda muito!
                transforms.ColorJitter(
                    brightness=0.3, 
                    contrast=0.3, 
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # Translação
                    scale=(0.9, 1.1)       # Zoom
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                ),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))  # Cutout
            ])
        
        elif augmentation_level == 'strong':
            # Augmentation forte - máximo desempenho
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.15
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.15, 0.15),
                    scale=(0.85, 1.15),
                    shear=10
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                ),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.15))
            ])
        
        else:
            raise ValueError(f"augmentation_level desconhecido: {augmentation_level}")
    
    else:
        # Apenas normalização para validação/teste
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
    
    return transform


def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    """
    Desnormaliza tensor para visualização
    
    Args:
        tensor: Tensor normalizado
        mean: Média usada na normalização
        std: Desvio padrão usado na normalização
        
    Returns:
        Tensor desnormalizado
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def plot_images(images, labels, predictions=None, num_images=16):
    """
    Plota grid de imagens com labels
    
    Args:
        images: Tensor de imagens
        labels: Labels verdadeiros
        predictions: Labels preditos (opcional)
        num_images: Número de imagens para plotar
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx in range(min(num_images, len(images))):
        # Desnormalizar imagem
        img = denormalize(images[idx].clone())
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # Plotar
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Título
        true_label = CIFAR10_CLASSES[labels[idx]]
        if predictions is not None:
            pred_label = CIFAR10_CLASSES[predictions[idx]]
            color = 'green' if labels[idx] == predictions[idx] else 'red'
            title = f'Real: {true_label}\nPred: {pred_label}'
        else:
            color = 'black'
            title = true_label
        
        axes[idx].set_title(title, color=color, fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    Plota gráficos de loss e acurácia durante treinamento
    
    Args:
        history: Dict com 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Treino', linewidth=2)
    ax1.plot(history['val_loss'], label='Validação', linewidth=2)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss durante Treinamento', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Acurácia
    ax2.plot(history['train_acc'], label='Treino', linewidth=2)
    ax2.plot(history['val_acc'], label='Validação', linewidth=2)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Acurácia (%)', fontsize=12)
    ax2.set_title('Acurácia durante Treinamento', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def load_image_for_prediction(image_path):
    """
    Carrega e pré-processa imagem para predição
    
    Args:
        image_path: Caminho para a imagem
        
    Returns:
        Tensor pronto para predição
    """
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Adiciona dimensão do batch
    return image_tensor, image


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Salva checkpoint do modelo
    
    Args:
        model: Modelo PyTorch
        optimizer: Otimizador
        epoch: Época atual
        loss: Loss atual
        path: Caminho para salvar
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint salvo em: {path}")


def load_checkpoint(model, optimizer, path):
    """
    Carrega checkpoint do modelo
    
    Args:
        model: Modelo PyTorch
        optimizer: Otimizador
        path: Caminho do checkpoint
        
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Checkpoint carregado da época {epoch}")
    return epoch, loss


class AverageMeter:
    """Calcula e armazena média e valor atual"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Calcula acurácia top-k
    
    Args:
        output: Predições do modelo
        target: Labels verdadeiros
        topk: Tuple com valores de k
        
    Returns:
        Lista de acurácias
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res
