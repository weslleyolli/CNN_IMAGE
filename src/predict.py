"""
Script para fazer predições com modelo treinado
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import create_model
from utils import (
    get_device,
    load_image_for_prediction,
    CIFAR10_CLASSES,
    get_transforms
)


class ImageClassifier:
    """Classe para classificar imagens"""
    
    def __init__(self, model_path='models/cnn_cifar10_best.pth', device=None):
        """
        Inicializa o classificador
        
        Args:
            model_path: Caminho do modelo treinado
            device: Device (CPU/GPU)
        """
        self.device = device if device else get_device()
        
        # Carregar modelo
        print(f"📦 Carregando modelo de: {model_path}")
        self.model = create_model(num_classes=10)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✓ Modelo carregado com sucesso!")
        
        self.transform = get_transforms(train=False)
    
    def predict(self, image_path, top_k=5):
        """
        Faz predição em uma imagem
        
        Args:
            image_path: Caminho da imagem ou objeto PIL Image
            top_k: Número de predições top-k para retornar
            
        Returns:
            Dict com predições
        """
        # Carregar e pré-processar imagem
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Transformar
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predição
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Top-k predições
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Resultados
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                'class': CIFAR10_CLASSES[idx],
                'class_id': int(idx),
                'probability': float(prob),
                'confidence': float(prob * 100)
            })
        
        return {
            'top_prediction': predictions[0],
            'all_predictions': predictions,
            'image': image
        }
    
    def visualize_prediction(self, result, save_path=None):
        """
        Visualiza resultado da predição
        
        Args:
            result: Resultado do método predict()
            save_path: Caminho para salvar imagem (opcional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Imagem
        ax1.imshow(result['image'])
        ax1.axis('off')
        ax1.set_title(
            f"Predição: {result['top_prediction']['class']}\n"
            f"Confiança: {result['top_prediction']['confidence']:.2f}%",
            fontsize=14,
            fontweight='bold'
        )
        
        # Gráfico de barras
        classes = [p['class'] for p in result['all_predictions']]
        confidences = [p['confidence'] for p in result['all_predictions']]
        
        colors = ['green' if i == 0 else 'skyblue' for i in range(len(classes))]
        
        ax2.barh(classes, confidences, color=colors)
        ax2.set_xlabel('Confiança (%)', fontsize=12)
        ax2.set_title('Top-5 Predições', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            ax2.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualização salva em: {save_path}")
        
        plt.show()
        
        return fig


def predict_image(image_path, model_path='models/cnn_cifar10_best.pth', visualize=True):
    """
    Função auxiliar para predição rápida
    
    Args:
        image_path: Caminho da imagem
        model_path: Caminho do modelo
        visualize: Se True, mostra visualização
        
    Returns:
        Dict com predições
    """
    classifier = ImageClassifier(model_path)
    result = classifier.predict(image_path)
    
    if visualize:
        classifier.visualize_prediction(result)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python predict.py <caminho_da_imagem>")
        print("Exemplo: python predict.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("="*70)
    print("CNN - CLASSIFICAÇÃO DE IMAGENS")
    print("="*70)
    print()
    
    result = predict_image(image_path)
    
    print("\n" + "="*70)
    print("RESULTADO DA PREDIÇÃO")
    print("="*70)
    print(f"\n🎯 Classe predita: {result['top_prediction']['class']}")
    print(f"📊 Confiança: {result['top_prediction']['confidence']:.2f}%")
    print("\nTop-5 predições:")
    for i, pred in enumerate(result['all_predictions'], 1):
        print(f"  {i}. {pred['class']:12s} - {pred['confidence']:5.2f}%")
    print()
