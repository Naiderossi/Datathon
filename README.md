# Datathon Decision — Plataforma de Matching de Talentos

Plataforma interativa em **Streamlit** para apoiar recrutadores na triagem de talentos. O app calcula **probabilidade de aderência candidato–vaga** combinando texto do currículo e metadados estruturados, e oferece ferramentas de **recomendação, predição e análise**.



## Link do streamlit  
https://datathondecisiontalents.streamlit.app/
---

## Sumário

* [Arquitetura](#arquitetura)
* [Justificativa do modelo preditivo](#justificativa-do-modelo-preditivo)
* [Principais funcionalidades](#principais-funcionalidades)
* [Como rodar localmente](#como-rodar-localmente)
* [Estrutura do repositório](#estrutura-do-repositório)
* [Datasets e modelos](#datasets-e-modelos)
* [Configurações (Kaggle/Secrets)](#configurações-kagglesecrets)
* [Dicas e troubleshooting](#dicas-e-troubleshooting)

---

## Arquitetura

**Frontend / UI (Streamlit, multi‑page)**

* Páginas em `datathon/pages/` e componentes principais em `datathon/talent_matching.py`.
* Estado leve via `st.session_state` (ex.: vaga selecionada na aba de sugestões reaproveitada no formulário).
* Cache de dados e de inferência com `st.cache_data`/`st.cache_resource` para ganho de desempenho.

**Camada de Negócio (matching e recomendação)**

* Pré‑processamento de texto (normalização, remoção de acentos, tokenização) → vetorização.
* **Redução de dimensionalidade (LSA/SVD)** e **classificador MLP** para estimar a probabilidade de match.
* Regras complementares: requisitos linguísticos (inglês/espanhol), formação mínima, sinais como PCD e SAP.
* Ordenação por score e aplicação de `threshold` configurável.

**Dados**

* Lidos de um **dataset Kaggle** (ou de arquivos locais), montando dataframes de *vagas*, *candidatos* e *prospects*.
* Normalização de colunas e índices (ex.: `job_id` como índice das vagas).

**Modelos e artefatos**

* `models/` contém os artefatos necessários em runtime:

  * `data_pipeline.joblib` — pipeline de features (TF‑IDF / LSA etc.)
  * `model_mlp_lsa.h5` — classificador MLP
  * `thresholds.json` — valores padrão de corte (ex.: `thr_f1`)

---

## Justificativa do modelo preditivo

Optamos por uma combinação **TF‑IDF/LSA (SVD)** para representar textos + um **classificador MLP** para estimar a probabilidade de aderência candidato–vaga. Abaixo, os motivos:

* **Compatível com dados tabulares + texto**: além dos vetores textuais, o MLP aceita bem sinais estruturados (ex.: PCD, SAP, níveis de idioma, formação), permitindo **interações não‑lineares** entre requisitos e atributos do candidato.
* **Bom custo‑benefício em CPU**: roda rápido em **Streamlit** sem GPU. O pipeline (TF‑IDF → LSA → MLP) tem **baixa latência** e footprint de memória reduzido, ideal para uso interativo.
* **Generalização com pouco dado**: para bases de porte pequeno/médio, **LSA** reduz ruído e melhora separabilidade; o **MLP** captura padrões além do linear sem exigir milhares de amostras rotuladas como transformers fine‑tuned.
* **Simplicidade de manutenção**: artefatos curtos (`joblib`/`h5`), re‑treino rápido e poucas dependências. Fácil de versionar e reimplantar.
* **Aderência ao domínio**: termos técnicos (ex.: SAP/ABAP/MM/SD) se beneficiam de LSA ao agrupar co‑ocorrências; o MLP modela **combinações** (ex.: *Inglês avançado + SAP + módulo MM*).
* **Threshold ajustável**: `thresholds.json` guarda cortes como `thr_f1`, possibilitando calibrar trade‑offs **precisão × recall** por negócio.

**Comparativos considerados**

* *Regressão Logística* / *SVM Linear*: fortes como baselines após TF‑IDF/LSA, porém capturam menos interações; em avaliações preliminares, o MLP apresentou melhor F1/PR‑AUC sob o mesmo pré‑processamento.
* *Árvores/Boosting*: funcionam bem em tabulares, mas exigem engenharia para integrar texto denso; tendem a crescer em tamanho e tempo de inferência.
* *Transformers (SBERT/e5) + Cross‑Encoder*: melhoram semântica e *recall* com embeddings, mas **custo operacional** e **latência** aumentam sem GPU. Mantivemos como **roadmap**.

**Quando considerar trocar o modelo**

* Volume grande, multilíngue ou alta ambiguidade semântica → adotar **sentence embeddings** (ex.: *e5‑base*, *all‑mpnet‑base‑v2*) com busca ANN (FAISS) + **re‑rank** por cross‑encoder.
* Forte exigência de explicabilidade → modelos lineares com coeficientes interpretáveis, *per‑feature*; manter MLP para produção e um linear como *shadow model* explicativo.
* Desejo de calibração probabilística estrita → aplicar **Platt scaling** ou **isotônica** após o MLP.

> Em resumo: **MLP + LSA** entrega um ótimo equilíbrio entre **qualidade**, **custo** e **manutenibilidade** neste cenário, mantendo espaço aberto para evoluir a semântica com embeddings/cross‑encoders conforme a necessidade e capacidade de infra.

---

## Principais funcionalidades

### 1) **Sugestão de Candidatos** (principal)

* **Um único seletor de vaga** baseado na base (sem upload de CSV nesta tela).
* Sliders para **threshold mínimo** e **limite de resultados**.
* Lista de candidatos recomendados com colunas úteis (idiomas, formação, sinais como PCD/SAP, *token overlap* e score).
* Expansor com **requisitos da vaga** e **descrição**.
* Ação para **download em CSV** da shortlist.

### 2) **Formulário e Predição**

* Formulário para compor/editar um perfil de candidato e **calcular o score de aderência** à vaga selecionada.
* Exibição dos fatores/atributos que mais impactam no resultado (explicabilidade básica).

### 3) **Panorama Executivo (opcional, se habilitado)**

* Visão de funil, contadores e tabelas de apoio para navegação rápida.

> Observação: A interface foi simplificada para evitar duplicidade de seletores e remover o upload de vagas nesta etapa. A aba **Sugestão de Candidatos** define a vaga; o **Formulário** reaproveita essa seleção.

---

## Como rodar localmente

### Pré‑requisitos

* **Python 3.10+** (3.11/3.12/3.13 também funcionam em CPU)
* `pip` e **virtualenv**/`venv`

### Passo a passo

1. **Clone** o repositório e crie o ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Datasets**:

   * **Opção A – Kaggle**: configure as credenciais (ver seção de [Configurações](#configurações-kagglesecrets)). O app baixa o dataset automaticamente na primeira execução.
   * **Opção B – Local**: coloque os arquivos equivalentes aos do dataset Kaggle em `datasets/` mantendo os **mesmos nomes**.
3. **Modelos**: garanta que `models/data_pipeline.joblib`, `models/model_mlp_lsa.h5` e `models/thresholds.json` existam. (Para testes, estes podem vir versionados ou disponibilizados por link interno.)
4. **Execute** o app:

   ```bash
   streamlit run app.py
   # Caso o ponto de entrada seja o diretório do app multipágina:
   # streamlit run datathon/Home.py
   ```
5. Abra `http://localhost:8501` no navegador.

### Atalhos úteis

* Recarregar o app: `R` no navegador quando o foco estiver na página do Streamlit.
* Limpar cache do Streamlit: menu ⋮ → *Clear cache* (ou apagar a pasta `.streamlit-cache` local).

---

## Estrutura do repositório

```text
.
├── datathon/
│   ├── Home.py                  # ponto de entrada (alternativo ao app.py)
│   ├── pages/
│   │   ├── 1_Formulario_e_Predicao.py
│   │   └── 2_Sugestao_de_Candidatos.py
│   ├── talent_matching.py       # regras de negócio e renderização
│   └── utils/                   # utilitários de pré-processamento etc.
├── models/
│   ├── data_pipeline.joblib
│   ├── model_mlp_lsa.h5
│   └── thresholds.json
├── datasets/                    # (opcional) dados locais equivalentes ao Kaggle
├── requirements.txt
└── app.py                       # ponto de entrada (se existir na raiz)
```

> A árvore acima é ilustrativa: ajuste conforme a estrutura real do repositório.

---

## Datasets e modelos

* **Dataset**: *DatathonDataset* (Kaggle). Utilize a mesma estrutura/nome dos arquivos ao optar por dados locais.
* **Artefatos de modelo** (CPU‑friendly): os arquivos em `models/` são carregados em runtime; TensorFlow/MLP roda em CPU sem dependências de GPU.

---

## Configurações (Kaggle/Secrets)

Você pode configurar as credenciais do Kaggle via **variáveis de ambiente** ou **`secrets.toml`** (recomendado em produção Streamlit Cloud).

### Variáveis de ambiente (local)

```bash
export KAGGLE_USERNAME="seu_usuario"
export KAGGLE_KEY="sua_chave"
```

### `.streamlit/secrets.toml`

```toml
[kaggle]
username = "seu_usuario"
key = "sua_chave"
```

> Observação: o código só lê os segredos **quando necessário** (no momento de baixar dados), evitando falhas durante import.

---
