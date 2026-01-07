"""
Aplicação Web Streamlit para Classificação de Imagens
CNN Melhorada com ResNet - 90.59% de acurácia
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os

from src.model_improved import create_improved_model
from src.utils import get_transforms, CIFAR10_CLASSES


# Configuração da página
st.set_page_config(
    page_title="CNN - Classificação de Imagens CIFAR-10",
    page_icon="🖼️",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Carrega o modelo melhorado (com cache)"""
    try:
        # Tentar carregar modelo melhorado primeiro
        model_path = 'models/cnn_improved_best.pth'
        if not os.path.exists(model_path):
            model_path = 'models/cnn_cifar10_best.pth'
        
        model = create_improved_model(model_type='improved', num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, True, model_path
    except Exception as e:
        return None, False, str(e)


def predict_image(image, model):
    """Faz predição na imagem"""
    # Pré-processar
    transform = get_transforms(train=False)
    image_tensor = transform(image).unsqueeze(0)
    
    # Predição
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Top-5
    top_probs, top_indices = torch.topk(probabilities, 5)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    results = []
    for idx, prob in zip(top_indices, top_probs):
        results.append({
            'class': CIFAR10_CLASSES[idx],
            'probability': float(prob),
            'confidence': float(prob * 100)
        })
    
    return results


def plot_predictions(predictions):
    """Cria gráfico de barras das predições"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = [p['class'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(classes))]
    
    bars = ax.barh(classes, confidences, color=colors)
    ax.set_xlabel('Confiança (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top-5 Predições', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Adicionar valores nas barras
    for i, (cls, conf) in enumerate(zip(classes, confidences)):
        ax.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


# Interface
def main():
    # Header
    st.title("🖼️ Classificação de Imagens com CNN")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Sobre o Projeto")
        st.markdown("""
        Este aplicativo usa uma **Rede Neural Convolucional (CNN)** 
        treinada no dataset **CIFAR-10** para classificar imagens em 10 categorias:
        
        - ✈️ Avião
        - 🚗 Automóvel
        - 🐦 Pássaro
        - 🐱 Gato
        - 🦌 Cervo
        - 🐕 Cachorro
        - 🐸 Sapo
        - 🐴 Cavalo
        - 🚢 Navio
        - 🚛 Caminhão
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Informações do Modelo")
        st.markdown("""
        - **Arquitetura**: ResNet Melhorado + SE Blocks
        - **Dataset**: CIFAR-10 (60.000 imagens)
        - **Acurácia**: 90.59% 🎯
        - **Parâmetros**: 11.17M
        - **Framework**: PyTorch
        - **Técnicas**: Mixup, Label Smoothing, Data Augmentation
        """)
    
    # Carregar modelo
    model, model_loaded, model_path = load_model()
    
    if not model_loaded:
        st.error(f"❌ Modelo não encontrado! Erro: {model_path}")
        st.info("Por favor, certifique-se de que o arquivo do modelo está em `models/cnn_improved_best.pth`")
        st.stop()
    
    st.success(f"✅ Modelo carregado: `{model_path}`")
    
    # Upload de imagem
    st.header("📤 Upload da Imagem")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Escolha uma imagem...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Formatos aceitos: JPG, JPEG, PNG, BMP"
        )
    
    with col2:
        st.info("""
        💡 **Dica**: O modelo foi treinado em imagens de 32x32 pixels 
        das categorias listadas na barra lateral. Para melhores resultados, 
        use imagens dessas categorias!
        """)
    
    # Imagens de exemplo
    with st.expander("🖼️ Não tem uma imagem? Use um exemplo!"):
        st.markdown("*Você pode baixar imagens de exemplo da internet ou usar suas próprias fotos*")
    
    if uploaded_file is not None:
        # Mostrar imagem
        image = Image.open(uploaded_file).convert('RGB')
        
        st.markdown("---")
        st.header("📊 Resultados")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Imagem Original")
            st.image(image, use_container_width=True)
            st.caption(f"Dimensões: {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            # Fazer predição
            with st.spinner("🔍 Analisando imagem..."):
                predictions = predict_image(image, model)
            
            # Resultado principal
            top_pred = predictions[0]
            
            st.subheader("🎯 Predição")
            st.markdown(f"### **{top_pred['class'].upper()}**")
            st.progress(top_pred['probability'])
            st.markdown(f"**Confiança: {top_pred['confidence']:.2f}%**")
            
            # Métricas
            st.markdown("---")
            
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Classe Predita", top_pred['class'])
            with metrics_cols[1]:
                st.metric("Confiança", f"{top_pred['confidence']:.1f}%")
            with metrics_cols[2]:
                st.metric("2ª Opção", predictions[1]['class'])
        
        # Gráfico de predições
        st.markdown("---")
        st.subheader("📈 Top-5 Predições")
        
        fig = plot_predictions(predictions)
        st.pyplot(fig)
        
        # Tabela detalhada
        with st.expander("📋 Ver todas as probabilidades"):
            st.table({
                'Posição': [f'{i+1}º' for i in range(5)],
                'Classe': [p['class'] for p in predictions],
                'Confiança (%)': [f"{p['confidence']:.2f}%" for p in predictions]
            })
    
    else:
        # Instruções
        st.info("👆 Faça upload de uma imagem para começar a classificação!")
        
        # Explicação do processo
        st.markdown("---")
        st.header("🔬 Como Funciona?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Upload
            Você faz upload de uma imagem no formato JPG, PNG ou BMP.
            """)
        
        with col2:
            st.markdown("""
            ### 2️⃣ Processamento
            A CNN analisa a imagem e extrai características visuais.
            """)
        
        with col3:
            st.markdown("""
            ### 3️⃣ Classificação
            O modelo retorna as 5 categorias mais prováveis com suas confianças.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Desenvolvido com ❤️ usando PyTorch e Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
