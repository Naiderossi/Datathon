# Datathon Decision - Plataforma de Matching

Plataforma interativa para apoiar recrutadores na triagem de talentos, avaliando aderência candidato/vaga com base em dados estruturados e texto do currículo. O app oferece visão executiva, predição de compatibilidade e recomendações de candidatos.

## Acesse o aplicativo

- [Abrir Datathon Decision no Streamlit](https://SEU-LINK-AQUI.streamlit.app/)  
  Substitua o link acima pela URL oficial publicada do projeto.

## Principais funcionalidades

- Panorama executivo com indicadores-chave e filtros rápidos sobre o funil de talentos.  
- Formulário inteligente para cadastrar ou editar candidatos e obter o score de compatibilidade em tempo real.  
- Recomendações de candidatos similares à vaga, com análise detalhada de currículo e competências.  
- Exportação de listas de candidatos para apoiar decisões do time de recrutamento.

## Como usar

### 1. Home
- Explore os cards superiores para acompanhar volume de candidatos, vagas e prospects ativos.  
- Ajuste os filtros laterais para atualizar gráficos e tabelas com dados segmentados.  
- Utilize as tabelas interativas para ordenar e pesquisar rapidamente informações relevantes.

### 2. Formulário e Predição
- Preencha os campos do candidato (experiência, idiomas, competências). Dados obrigatórios aparecem com indicação visual.  
- Use as sugestões automáticas de skills para acelerar o preenchimento; basta selecionar na lista exibida.  
- Clique em **Calcular aderência** para gerar o score e visualizar as competências que mais impactaram o resultado.  
- Consulte as recomendações de vagas relacionadas geradas na seção inferior.

### 3. Sugestão de Candidatos
- Carregue uma vaga usando JSON/CSV no painel lateral ou selecione uma vaga já listada.  
- Aplique filtros (nível profissional, PCD, disponibilidade) para refinar o conjunto de talentos sugeridos.  
- Abra o perfil de cada candidato para visualizar currículo, certificações e histórico de recomendações.  
- Utilize o botão de exportação para baixar a shortlist filtrada em CSV.

## Dados utilizados

- Os datasets completos ultrapassam 25 MB e não são versionados no repositório. O app carrega essas informações a partir do armazenamento configurado no deploy.  
- Para demonstrações ou testes rápidos, utilize os arquivos de `datasets/samples/`, que contêm subconjuntos leves dos dados reais.  
- Sempre que os dados brutos forem atualizados, execute `python scripts/create_sample_datasets.py` para atualizar as amostras.

## Execução local (opcional)

1. Crie um ambiente virtual (`python -m venv .venv`) e instale as dependências com `pip install -r requirements.txt`.  
2. Posicione os datasets completos na pasta `datasets/` (fora do versionamento) ou reutilize as amostras de `datasets/samples/`.  
3. Rode `streamlit run app.py` na raiz do projeto e acesse `http://localhost:8501` no navegador.  
4. Para desempenho ideal, mantenha os artefatos do modelo em `models/` (`data_pipeline.joblib`, `model_mlp_lsa.h5`, `thresholds.json`).

## Suporte

- Dúvidas sobre o uso do aplicativo podem ser registradas via issues no GitHub.  
- Ajustes no modelo ou pipelines devem seguir os scripts em `src/` e `scripts/`, como documentado internamente para a equipe técnica.
