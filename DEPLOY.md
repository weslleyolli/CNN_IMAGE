# 🚀 Guia de Deploy no Streamlit Cloud

## ✅ Passo 1: Código já está no GitHub
✓ Repositório: https://github.com/weslleyolli/CNN_IMAGE.git
✓ Branch: main
✓ Arquivos necessários prontos

## 📋 Passo 2: Deploy no Streamlit Cloud

### 1. Acesse o Streamlit Cloud
🔗 https://streamlit.io/cloud

### 2. Faça Login
- Clique em "Sign in"
- Use sua conta GitHub

### 3. Deploy da Aplicação
1. Clique em **"New app"**
2. Preencha os campos:
   - **Repository**: `weslleyolli/CNN_IMAGE`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: escolha um nome (ex: `cnn-image-classifier`)

3. Clique em **"Deploy!"**

### 4. Aguarde o Deploy
- ⏱️ Tempo estimado: 5-10 minutos
- O Streamlit vai:
  - Instalar dependências do `requirements.txt`
  - Carregar os modelos (45MB)
  - Iniciar a aplicação

## 🎯 Resultado

Sua aplicação estará disponível em:
```
https://cnn-image-classifier-seu-nome.streamlit.app
```

## ⚙️ Configurações Avançadas (Opcional)

### Aumentar Recursos
Se a aplicação ficar lenta:

1. No painel do Streamlit Cloud, vá em **Settings**
2. **Advanced settings** → **Python version**: 3.11
3. **Resources**: Se disponível, aumente memória

### Variáveis de Ambiente
Se precisar (não necessário para este projeto):
```
Settings → Secrets → Add secrets
```

## 🔧 Solução de Problemas

### Erro: "File size too large"
Se os modelos forem muito grandes:

1. **Opção A**: Usar Git LFS
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Add Git LFS"
git push
```

2. **Opção B**: Baixar modelo no primeiro uso
Modifique `app.py` para baixar de um link externo.

### Erro: "Module not found"
Certifique-se que `requirements.txt` tem todas as dependências.

### Aplicação Lenta
- Modelos grandes (~45MB) podem demorar no primeiro carregamento
- Use `@st.cache_resource` para cachear modelo (já implementado ✓)

## 📊 Monitoramento

Após deploy:
- **Logs**: Settings → View logs
- **Analytics**: Veja quantas pessoas usam
- **Usage**: Monitorar recursos

## 🔄 Atualizações

Para atualizar a aplicação:
```bash
# Faça mudanças no código
git add .
git commit -m "Descrição das mudanças"
git push origin main
```

O Streamlit Cloud fará redeploy automático! 🎉

## 💡 Dicas

1. **Teste local primeiro**: `streamlit run app.py`
2. **Otimize modelos**: Considere quantização se muito grande
3. **Cache tudo**: Use `@st.cache_data` e `@st.cache_resource`
4. **Mobile-friendly**: Teste em celular

## 🌐 Compartilhar

Depois do deploy:
- ✅ Compartilhe o link com qualquer pessoa
- ✅ Adicione ao README.md do GitHub
- ✅ Compartilhe no LinkedIn
- ✅ Adicione ao portfólio

---

## 🎊 Projeto Completo!

✅ Código no GitHub
✅ Modelo treinado (90.59%)
✅ Pronto para deploy
✅ Documentação completa

**Próximo passo**: Acesse https://streamlit.io/cloud e faça o deploy! 🚀
