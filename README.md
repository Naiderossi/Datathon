# Datathon Decision - Plataforma de Matching

Plataforma Streamlit para apoiar recrutadores na triagem de talentos, avaliando a aderência candidato/vaga com base em dados estruturados e texto do currículo, além de sugerir candidatos e oportunidades similares. O projeto inclui pipeline completo: pré-processamento, feature engineering, treinamento do modelo MLP + LSA e painel analítico.

## Estrutura do repositório

- `app.py` – dashboard executivo (home) com visão geral dos datasets
- `pages/`
  - `1_Formulario_e_Predicao.py` – formulário padronizado + predição do candidato
  - `2_Sugestao_de_Candidatos.py` – módulo de sourcing e recomendações de talentos
- `talent_matching.py` – lógica compartilhada entre as páginas (triagem e sugestões)
- `src/` – utilitários, pré-processamento, engenharia de atributos, treino e avaliação
- `models/` – artefatos serializados (`data_pipeline.joblib`, `model_mlp_lsa.h5`, `thresholds.json`…)
- `datasets/` – dados brutos (fora do versionamento) e amostras leves para demos
- `tests/` – ponto de partida para testes automatizados
- `notebooks/` – espaço reservado para análises exploratórias e experimentos
- `backup/` – versões anteriores de scripts de interface (referência)

## Requisitos

- Python 3.10 ou 3.11
- Bibliotecas listadas em `requirements.txt`

Instalação recomendada:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Executando a aplicação

1. Garanta que os arquivos necessários estejam em `datasets/` e os artefatos do modelo em `models/`.
2. Ative o ambiente virtual (conforme acima).
3. Rode a interface multipágina pela raiz do projeto:

```bash
streamlit run app.py
```

Opcionalmente, `streamlit run app2.py` carrega a página de triagem direta por compatibilidade com versões antigas.

- A página “Home” apresenta indicadores, gráficos e tabelas com insights dos três CSVs.
- A página “Formulário e Predição” reúne o formulário padronizado com sugestões automáticas de competências, predição online e recomendações de vagas relacionadas.
- A página “Sugestão de Candidatos” permite carregar vagas (JSON ou CSV), filtrar candidatos livres, visualizar currículos e exportar shortlists.

## Treinando o modelo novamente

O módulo `src/train.py` concentra o fluxo de treinamento (TF-IDF + SVD + MLP). Para recriar os artefatos padrão na pasta `models/`:

```bash
python -c "from pathlib import Path; from src.train import train_model; train_model(Path('models'))"
```

- Utilize os CSVs rotulados presentes em `datasets/`.
- O treinamento salva `data_pipeline.joblib`, `model_mlp_lsa.h5`, `thresholds.json`, `registry.json` e `training_report.json`.
- Ajustes adicionais de features podem ser feitos nos módulos `src/preprocessing.py` e `src/feature_engineering.py`.

## Justificativa do modelo de predição

- **Combinação texto + numérico**: o modelo precisa capturar requisitos em linguagem natural e comparar com atributos estruturados dos candidatos. O pipeline TF-IDF + LSA transforma descrições e currículos em vetores densos, preservando semântica mesmo com variações de idioma.
- **Redução de dimensionalidade**: a decomposição SVD (LSA) reduz ruído e torna o treinamento estável em relação a sinônimos ou termos raros, importante para currículos heterogêneos.
- **Modelagem não linear**: a MLP (com BatchNorm) aprende interações entre sinais textuais e numéricos (níveis de idioma, PCD, SAP, sobreposição de skills), alcançando melhor desempenho que modelos lineares puros.
- **Eficiência operacional**: a arquitetura mantém inferência rápida (matrizes densas + rede rasa) e reaproveita o mesmo `data_pipeline.joblib`, simplificando deploy no Streamlit.
- **Interpretabilidade controlada**: as colunas derivadas (diferença de níveis, overlaps) permanecem disponíveis para explicar o score, enquanto o componente neural agrega ganho de acurácia.

## Testes e validação

- Sintaxe rápida: `python -m py_compile app.py pages/1_Formulario_e_Predicao.py pages/2_Sugestao_de_Candidatos.py`
- Adicione testes unitários em `tests/` conforme novas funcionalidades forem criadas.

## Dados

- Os arquivos completos em `datasets/` ultrapassam o limite de 25 MB do GitHub e devem ficar fora do controle de versão (estão ignorados pelo `.gitignore`).
- O diretório `datasets/samples/` armazena subconjuntos leves gerados a partir dos dados reais para desenvolvimento local e demonstrações.
- Recrie as amostras executando `python scripts/create_sample_datasets.py` sempre que atualizar os dados brutos.
- Para rodar o app com o volume integral, mantenha os arquivos originais na raiz de `datasets/` com os mesmos nomes utilizados internamente.

## Observações

- Os datasets são volumosos; execute o app localmente utilizando o cache do Streamlit (`@st.cache_data`) já configurado.
- Para atualizar vagas ou artefatos em produção (Streamlit Cloud), mantenha o layout de diretórios descrito acima.
