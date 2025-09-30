# Datathon Decision — Plataforma de Matching de Talentos

Plataforma interativa em **Streamlit** para apoiar recrutadores na triagem de talentos. O app calcula **probabilidade de aderência candidato–vaga** combinando texto do currículo e metadados estruturados, e oferece ferramentas de **recomendação, predição e análise**.

---

## Links

### Link do streamlit

[https://datathondecisiontalents.streamlit.app/](https://datathondecisiontalents.streamlit.app/)

---

### Link do vídeo de apresentação

[https://drive.google.com/file/d/1rbzG1PEb0pQSH0CfTkMQBIrf6BPLf4mN/view?usp=sharing](https://drive.google.com/file/d/1rbzG1PEb0pQSH0CfTkMQBIrf6BPLf4mN/view?usp=sharing)

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


### Métricas de referência

>  

| Métrica    | Valor | Observações                                       |
| ---------- | ----- | ------------------------------------------------- |
| Acurácia   | 65.7% | Test set (ex.: hold‑out 20% estratificado)        |
| F1 (macro) | 64.3% | Threshold calibrado em `thr_f1` (thresholds.json) |

**Notas**

* Threshold ótimo (`best_threshold`): **0.3365**
* F1 (classe positiva) no threshold ótimo: **71.4%**; F1 @ 0.50: **65.7%**
* Acurácia global: **65.7%** (983 amostras no teste)

**Metodologia sugerida**

* Divisão **estratificada** em 80/20 (ou **5‑fold CV**), com *seed* fixa para reprodutibilidade.
* Calibração do threshold para **F1** (valor salvo em `models/thresholds.json → mlp_thresholds.thr_f1`).
* Reportar métricas **apenas** no conjunto de teste (nunca no treino).

---

## Principais funcionalidades

### 1) **Sugestão de Candidatos** (principal)

* **Um único seletor de vaga** baseado na base (sem upload de CSV nesta tela para reduzir processamento de página).
* Sliders para **compatibilidade mínima** e **limite de resultados**.
* Lista de candidatos recomendados com colunas úteis (idiomas, formação, sinais como PCD/SAP, *token overlap* e score).
* Expansor com **requisitos da vaga** e **descrição**.
* Ação para **download em CSV** da shortlist.
* Possibilidade de melhorias utilizando automatização de envio de agendamentos de entrevistas.

### 2) **Formulário e Predição**

* Formulário para compor/editar um perfil de candidato e **calcular o score de aderência** à vaga selecionada, pensando em uma entrevista ou analise de candidato guiada, a fim de padronizar os curriculos internos.
* Exibição dos fatores/atributos que mais impactam no resultado (explicabilidade básica).

### 3) **Panorama Executivo (opcional, se habilitado)**

* Visão de funil, contadores e tabelas de apoio para navegação rápida.

> Observação: A interface foi simplificada para evitar duplicidade de seletores e remover o upload de vagas nesta etapa. A aba **Sugestão de Candidatos** define a vaga; o **Formulário** reaproveita essa seleção.

---

## Como rodar localmente

### Pré‑requisitos

* **Python 3.10+** 
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
   * **Opção B – Local**: coloque os arquivos equivalentes aos do dataset Kaggle em `datasets/` mantendo os **mesmos nomes** : df_applicants.csv, df_jobs.csv, df_prospects.csv.
3. **Modelos**: garanta que `models/data_pipeline.joblib`, `models/model_mlp_lsa.h5` e `models/thresholds.json` existam. 
4. **Execute** o app:

   ```bash
   streamlit run app.py
   # Caso o ponto de entrada seja pelo app principal:
   # streamlit run datathon/app.py
   # Caso o ponto de entrada seja diretamente pela sugestão de candidatos:
   # streamlit run datathon/app2.py
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
│   ├── src/            # utilitários de pré-processamento etc.
|   |   ├── preprossesing.py
|   |   ├── feature_engineering.py
|   |   ├── evaluate.py
|   |   ├── mlp_infer.py
|   |   ├── model_utils.py
|   |   ├── utils.py
|   |   ├── train.py        
│   ├── pages/
│   │   ├── 1_Formulario_e_Predicao.py
│   │   └── 2_Sugestao_de_Candidatos.py                  
|   ├── models/
│       ├── data_pipeline.joblib
│       ├── model_mlp_lsa.h5
│       └── thresholds.json
|       └── training_report.json         
├── requirements.txt
└── app.py                   # principal
└── app2.py                  # alteranativa para rodar direto a sugestão de candidatos
└── train_mlp.py
└── talent_matching.py       # regras de negócio e renderização                     
```


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
## Dicas e possíveis problemas 

* **Depreciação ************************************`use_container_width`**: prefira `width="stretch"` (ou `"content"`).
* **TensorFlow CPU**: logs informam otimizações por instruções da CPU; não é erro.
* **Colunas ausentes**: se atualizar os CSVs, mantenha colunas esperadas (`job_id`, `titulo`, campos de idiomas etc.).
* **Falhas ao buscar recomendações**: verifique se os artefatos em `models/` existem e se o dataset possui candidatos/vagas suficientes.
* **Crash da aplicação (datasets e data_pipeline exigem muito processamento e podem ultrapasar a cota do streamlit cloud), basta rebootar o app. Maiores análises e melhorias de perfomance podém ser executadas em outros sprints, reduzindo datasets, utilizando arquivos padronizados.. etc
