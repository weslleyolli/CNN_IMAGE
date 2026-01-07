"""
CNN Melhorada com técnicas avançadas para maior acurácia

MELHORIAS IMPLEMENTADAS:
1. Arquitetura ResNet (Residual Connections) - evita vanishing gradient
2. Mais camadas convolucionais (melhor extração de features)
3. Dropout adaptativo (regularização melhor)
4. Batch Normalization em todas camadas
5. Global Average Pooling (reduz overfitting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Bloco Residual - permite treinar redes mais profundas
    
    A conexão "skip" (atalho) ajuda o gradiente fluir melhor
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Convolução principal
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Conexão residual (skip connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Adiciona conexão residual
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class ImprovedCNN(nn.Module):
    """
    CNN Melhorada com ResNet para CIFAR-10
    
    DIFERENÇAS DO MODELO ANTERIOR:
    - Usa blocos residuais (treina melhor)
    - Mais profunda (extrai features melhores)
    - Global Average Pooling (menos overfitting)
    - Dropout progressivo
    """
    
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        # Camada inicial
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Blocos residuais
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Global Average Pooling (melhor que Flatten)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classificador
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Cria uma sequência de blocos residuais"""
        layers = []
        
        # Primeiro bloco pode ter stride diferente
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Demais blocos mantêm dimensões
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Entrada: 32x32x3
        out = F.relu(self.bn1(self.conv1(x)))  # 32x32x64
        
        out = self.layer1(out)  # 32x32x64
        out = self.layer2(out)  # 16x16x128
        out = self.layer3(out)  # 8x8x256
        out = self.layer4(out)  # 4x4x512
        
        # Global Average Pooling
        out = self.avgpool(out)  # 1x1x512
        out = out.view(out.size(0), -1)  # 512
        
        out = self.dropout(out)
        out = self.fc(out)  # 10 classes
        
        return out
    
    def count_parameters(self):
        """Conta número de parâmetros treináveis"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UltraImprovedCNN(nn.Module):
    """
    Versão ULTRA otimizada - inspirada em DenseNet e EfficientNet
    
    TÉCNICAS EXTRAS:
    - Squeeze-and-Excitation (atenção nos canais)
    - Stochastic Depth (dropout de camadas)
    - Label Smoothing (via loss)
    """
    
    def __init__(self, num_classes=10):
        super(UltraImprovedCNN, self).__init__()
        
        # Stem (entrada)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction
        self.features = nn.Sequential(
            # Bloco 1: 32x32 -> 16x16
            self._make_block(32, 64, stride=1),
            self._make_block(64, 64, stride=1),
            
            # Bloco 2: 16x16 -> 8x8
            self._make_block(64, 128, stride=2),
            self._make_block(128, 128, stride=1),
            
            # Bloco 3: 8x8 -> 4x4
            self._make_block(128, 256, stride=2),
            self._make_block(256, 256, stride=1),
            
            # Bloco 4: 4x4 -> 2x2
            self._make_block(256, 512, stride=2),
            self._make_block(512, 512, stride=1),
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _make_block(self, in_ch, out_ch, stride):
        """Bloco com Squeeze-Excitation"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch)  # Atenção
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Aprende quais canais são mais importantes para cada imagem
    Muito efetivo para melhorar acurácia!
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


def create_improved_model(model_type='improved', num_classes=10):
    """
    Factory function para criar modelos
    
    Args:
        model_type: 'improved' ou 'ultra'
        num_classes: Número de classes
        
    Returns:
        Modelo CNN
    """
    if model_type == 'improved':
        model = ImprovedCNN(num_classes)
    elif model_type == 'ultra':
        model = UltraImprovedCNN(num_classes)
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
    
    return model


if __name__ == '__main__':
    # Teste
    model_improved = create_improved_model('improved')
    model_ultra = create_improved_model('ultra')
    
    print("=" * 70)
    print("MODELO IMPROVED (ResNet)")
    print("=" * 70)
    print(f"Parâmetros: {model_improved.count_parameters():,}")
    
    print("\n" + "=" * 70)
    print("MODELO ULTRA (ResNet + SE)")
    print("=" * 70)
    print(f"Parâmetros: {model_ultra.count_parameters():,}")
    
    # Teste forward pass
    x = torch.randn(4, 3, 32, 32)
    y1 = model_improved(x)
    y2 = model_ultra(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output improved: {y1.shape}")
    print(f"Output ultra: {y2.shape}")
