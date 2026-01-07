# 🖼️ CNN para Classificação de Imagens - CIFAR-10

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)
![Acurácia](https://img.shields.io/badge/Acurácia-90.59%25-success.svg)

**Rede Neural Convolucional (CNN)** com arquitetura ResNet melhorada para classificação de imagens do dataset CIFAR-10, atingindo **90.59% de acurácia**.

## 🎯 Sobre o Projeto

Este projeto implementa uma CNN de última geração para classificar imagens em 10 categorias:
- ✈️ Avião | 🚗 Automóvel | 🐦 Pássaro | 🐱 Gato | 🦌 Cervo
- 🐕 Cachorro | 🐸 Sapo | 🐴 Cavalo | 🚢 Navio | 🚛 Caminhão

## 📊 Resultados

| Modelo | Acurácia | Épocas | Arquitetura |
|--------|----------|--------|-------------|
| **CNN Melhorado** | **90.59%** | 30 | ResNet + SE Blocks |
| CNN Básico | 85.57% | 50 | Conv + MaxPool |

## 🚀 Tecnologias

- **PyTorch** - Framework de Deep Learning
- **Streamlit** - Interface web interativa
- **Matplotlib** - Visualização de dados

## 🏗️ Arquitetura do Modelo

### Técnicas Implementadas

✅ **Residual Connections** - Permite treinar redes mais profundas  
✅ **Squeeze-and-Excitation Blocks** - Mecanismo de atenção  
✅ **Data Augmentation** - Rotação, zoom, flip, cutout  
✅ **Mixup** - Mistura de imagens durante treinamento  
✅ **Label Smoothing** - Reduz overconfidence  
✅ **Cosine Annealing LR** - Scheduler de learning rate  

## 📦 Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/CNN_image.git
cd CNN_image

# Crie ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

## 🎮 Como Usar

### Interface Web
```bash
streamlit run app.py
```

### Treinar Modelo
```bash
python src/train_improved.py --epochs 30
```

## 📁 Estrutura

```
CNN_image/
├── app.py                    # Aplicação Streamlit
├── src/
│   ├── model_improved.py    # CNN Melhorada
│   ├── train_improved.py    # Treinamento
│   └── utils.py             # Funções auxiliares
├── models/                   # Modelos treinados
└── data/                     # Dataset
```

## 📚 Referências

- **ResNet**: [Deep Residual Learning (He et al., 2016)](https://arxiv.org/abs/1512.03385)
- **Mixup**: [Beyond Empirical Risk Minimization (Zhang et al., 2018)](https://arxiv.org/abs/1710.09412)
- **SE-Net**: [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)

## 👨‍💻 Autor

Desenvolvido com ❤️ por **Weslley Oliveira**

---

**Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) - Alex Krizhevsky, 2009
