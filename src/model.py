"""
Arquitetura da CNN para classificação CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    Rede Neural Convolucional para classificar imagens do CIFAR-10
    
    Arquitetura:
    - 3 blocos convolucionais (Conv → ReLU → MaxPool)
    - 2 camadas fully connected
    - Dropout para regularização
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        # Bloco Convolucional 1: 3 → 32 canais
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 → 16x16
        
        # Bloco Convolucional 2: 32 → 64 canais
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 → 8x8
        
        # Bloco Convolucional 3: 64 → 128 canais
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 → 4x4
        
        # Camadas Fully Connected
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Forward pass da rede
        
        Args:
            x: Tensor de entrada (batch_size, 3, 32, 32)
            
        Returns:
            Tensor de saída (batch_size, num_classes)
        """
        # Bloco 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Bloco 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Bloco 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 128*4*4)
        
        # Fully Connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def count_parameters(self):
        """Conta o número total de parâmetros treináveis"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes=10, pretrained=False):
    """
    Factory function para criar o modelo
    
    Args:
        num_classes: Número de classes (padrão: 10 para CIFAR-10)
        pretrained: Se True, carrega pesos pré-treinados
        
    Returns:
        Modelo CNN
    """
    model = CIFAR10CNN(num_classes=num_classes)
    
    if pretrained:
        # Carregar pesos salvos (se existirem)
        try:
            model.load_state_dict(torch.load('models/cnn_cifar10.pth'))
            print("✓ Modelo pré-treinado carregado com sucesso!")
        except FileNotFoundError:
            print("⚠ Arquivo de modelo não encontrado. Usando pesos aleatórios.")
    
    return model


if __name__ == "__main__":
    # Teste rápido do modelo
    model = create_model()
    print(f"Modelo criado com {model.count_parameters():,} parâmetros")
    
    # Teste com tensor dummy
    dummy_input = torch.randn(4, 3, 32, 32)  # batch de 4 imagens
    output = model(dummy_input)
    print(f"Shape de entrada: {dummy_input.shape}")
    print(f"Shape de saída: {output.shape}")
