# Mortes nas rodovias federais brasileiras: onde, quando e por quê (PRF 2017–2025)

Projeto final do curso de **Análise de Dados (EBAC × Semantix)**, por **Thiago Paiano**.

📊 **Painel no Looker Studio:** LINK_DO_PAINEL
📓 **Análise exploratória:** [`notebooks/analise_exploratoria.ipynb`](notebooks/analise_exploratoria.ipynb)

---

## 1. O problema e sua relevância

Entre 2017 e 2025, **50.822 pessoas morreram** e **167.379 ficaram gravemente feridas** em acidentes nas rodovias federais registrados pela Polícia Rodoviária Federal (PRF). Sinistros de trânsito estão entre as principais causas de morte de jovens no Brasil e geram custos altos para o SUS, a Previdência e a economia: perda de produtividade, atendimento hospitalar e reabilitação.

O Brasil assumiu o compromisso da **Segunda Década de Ação pela Segurança no Trânsito (ONU/OMS, 2021–2030)** e do **PNATRANS** (Plano Nacional de Redução de Mortes e Lesões no Trânsito), cuja meta é **reduzir as mortes pela metade até 2030**. Os dados da PRF mostram que, nas rodovias federais, o país está longe disso: as mortes de 2025 (6.043) estão praticamente no mesmo patamar de 2017 (6.248).

**Pergunta de negócio:** *onde, quando e em que tipo de acidente as pessoas morrem nas rodovias federais, e quais ações teriam mais impacto para reduzir essas mortes?*

## 2. Coleta de dados

| Item | Detalhe |
|---|---|
| Fonte primária | [Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf), conjunto *"Acidentes — agrupados por ocorrência"* (DATATRAN) |
| Período | 01/01/2017 a 31/12/2025, 9 arquivos anuais |
| Volume | 632.667 ocorrências × 30 colunas |
| Granularidade | 1 linha = 1 acidente (data, hora, UF, BR, km, município, causa, tipo, classificação, fase do dia, clima, tipo de pista, traçado, uso do solo, nº de mortos/feridos/ilesos/veículos, coordenadas) |
| Licença | Dados abertos do Governo Federal |

**Método de coleta:** os arquivos `datatranAAAA.zip` foram obtidos do portal da PRF; para reprodutibilidade, usou-se a cópia dos arquivos originais publicada no repositório [thaleraaa/analiseACIDENTE-RODOVIAS](https://github.com/thaleraaa/analiseACIDENTE-RODOVIAS) (pasta `dados/bronze/PRF`). Validação feita:
- nenhum `id` duplicado;
- todos os anos completos, de 1º de janeiro a 31 de dezembro;
- totais anuais conferidos.

Os dados brutos (~180 MB) não foram versionados. Para reproduzir:

```bash
pip install pandas pyarrow statsmodels scikit-learn matplotlib jupyter
# baixe datatran2017.csv ... datatran2025.csv para uma pasta, ex.: dados/brutos/
python src/preparar_dados.py dados/brutos
jupyter notebook notebooks/analise_exploratoria.ipynb
```

### Limpeza e tratamento (`src/preparar_dados.py`)
- **Leitura:** arquivos em `latin1`, separador `;`, decimal com vírgula (exceto 2024, com ponto), convertidos de forma única.
- **Harmonização de causas:** em 2021 a PRF trocou a taxonomia. Por exemplo, *"Falta de Atenção à Condução"* foi substituída por *"Reação tardia ou ineficiente do condutor"* e *"Ausência de reação do condutor"*, e *"Ingestão de Álcool"* virou *"Ingestão de álcool pelo condutor"*. As **91 descrições** foram mapeadas em **12 macrogrupos** comparáveis ao longo do tempo: álcool/drogas, velocidade, desatenção/reação tardia, ultrapassagem/contramão, desrespeito a normas, sono/mal súbito, pedestre, veículo, via/infraestrutura, clima, animais e outros.
- **Harmonização de tipos:**
  - *Colisão lateral mesmo sentido / sentido oposto* → *Colisão lateral*
  - *Colisão com objeto estático / em movimento* → *Colisão com objeto*
  - tipos residuais → *Outros*
- **Variáveis derivadas:** ano, mês, hora, dia da semana, fim de semana, `fatal` (ao menos 1 morto), `urbano` (`uso_solo = Sim`), `noite`.
- **Qualidade:**
  - coordenadas fora do território brasileiro (54) → nulas
  - clima "Ignorado" → nulo (1,4%)
  - `tracado_via` com múltiplos valores → primeiro valor

## 3. Modelagem

### 3.1 Modelagem dos dados
```
datatranAAAA.csv (bruto, 9 arquivos)
        │  preparar_dados.py: limpeza, tipagem, harmonização
        ▼
acidentes_limpo.parquet  ← tabela fato (1 linha = 1 acidente, 632.667 linhas)
        │  agregações
        ├── looker_acidentes_agregado.csv  (ano × UF × tipo × causa × fase do dia × pista × urbano → acidentes, fatais, mortos, feridos)  → fonte do painel
        ├── agg_br.csv                      (trecho BR/UF → acidentes, mortos, letalidade)  → fonte do painel
        └── agg_ano / agg_tipo / agg_causa / agg_uf_ano / agg_hora_fase / agg_dia_hora / agg_condicoes
```
O painel usa tabelas **pré-agregadas** (135 mil linhas em vez de 632 mil), o que o deixa leve, mas o mantém filtrável. A métrica central é a **letalidade**: `mortos ÷ acidentes × 100` (mortos a cada 100 acidentes). Ela é recalculada no Looker como campo calculado, para continuar correta sob qualquer filtro.

### 3.2 Análise exploratória
O notebook cobre:
- qualidade dos dados
- evolução anual
- tipo de acidente
- causas
- hora, fase do dia e fim de semana
- UF e trechos de BR
- condições da via e do clima
- matriz de correlação

### 3.3 Modelo estatístico: regressão logística
Para separar efeitos que se sobrepõem (por exemplo, trecho rural costuma ser pista simples), foi ajustada uma **regressão logística** (`statsmodels`).
- **Alvo:** `fatal`.
- **Explicativas:**
  - tipo de pista
  - rural/urbano
  - fase do dia
  - fim de semana
  - tipo de acidente
  - grupo de causa
- **Validação:** separação treino/teste 70/30.
- **Resultado:** **AUC = 0,78** no teste; pseudo-R² de McFadden = 0,16.

| Fator (vs. referência) | Razão de chance (OR) |
|---|---|
| Atropelamento de pedestre (vs. colisão traseira) | **6,6** |
| Colisão frontal (vs. colisão traseira) | **5,4** |
| Trecho rural (vs. urbano) | **2,1** |
| Amanhecer / plena noite (vs. pleno dia) | **2,0 / 1,8** |
| Causa: velocidade incompatível (vs. desatenção) | **1,7** |
| Causa: ultrapassagem/contramão (vs. desatenção) | **1,6** |
| Pista simples (vs. dupla/múltipla) | **1,5** |
| Fim de semana (vs. dia útil) | **1,2** |

*Todos com p < 0,001.* O modelo é explicativo (associação), não causal. A causa *álcool* aparece com OR < 1, o que provavelmente é efeito de registro: o teste do condutor é feito com mais frequência quando ele sobrevive.

## 4. Principais insights

1. **As mortes não estão caindo.** Foram 5.292 mortes em 2020 (o mínimo da série) e 6.160 em 2024 / 6.043 em 2025. A queda no número de acidentes após 2017 está concentrada nos acidentes **sem vítimas** (30,7 mil → ~11 mil), o que indica mudança de registro e não estradas mais seguras. A letalidade subiu de 7,0 para cerca de 8,3 mortos por 100 acidentes.
2. **Dois tipos de acidente concentram quase metade das mortes.** Colisão frontal e atropelamento de pedestre são apenas **11%** dos acidentes, mas **47%** das mortes. A colisão frontal mata 37,9 a cada 100 acidentes, contra 4,5 na colisão traseira.
3. **Pista simples e trecho rural.** Metade dos acidentes ocorre em pista simples, mas **70% das mortes**: 11,3 mortos por 100 acidentes, contra 4,9 na pista dupla. Trechos rurais têm letalidade 2,2× maior que os urbanos.
4. **A noite mata mais.** Plena noite reúne 35% dos acidentes e **48% das mortes**. A letalidade às 3h (14,1) é 3× a das 8h (4,7). No fim de semana, a letalidade é 31% maior.
5. **Causas mais letais:**
   - pedestre (29,5 mortos/100)
   - ultrapassagem/contramão (16,4)
   - sono/mal súbito (11,8)
   - velocidade (10,7)

   *Desatenção* é a causa mais frequente (38% dos acidentes).
6. **Geografia.**
   - **Volume:** MG (6.637 mortos), PR (5.033) e BA (4.829).
   - **Letalidade:** MA (19,9 por 100, 2,5× a média nacional de 8,0), PA (16,9), TO (14,4) e BA (14,2).
   - **Trechos com mais mortes:** BR-381/MG, BR-116/SP, BR-277/PR, BR-101/BA e BR-040/MG.
   - Santa Catarina tem muitos acidentes e baixa letalidade (4,7); a BR-101/SC, duplicada, tem 3,1.
7. **O clima pesa pouco.** A chuva tem letalidade *menor* que céu claro (7,2 vs 8,4), provavelmente porque se dirige mais devagar. A exceção é a **neblina** (13,6).

## 5. Ações sugeridas

| Prioridade | Ação | Evidência |
|---|---|---|
| 1 | **Duplicação e separação física de sentidos** (barreira central, faixas adicionais de ultrapassagem — "2+1") nos trechos de pista simples com mais mortes, começando por BR-101/BA, BR-116/BA e MG, BR-381/MG, BR-010/MA e PA | Colisão frontal = 31% das mortes; pista simples OR 1,5; SC duplicada com letalidade de 3,1 |
| 2 | **Fiscalização de ultrapassagem e velocidade** com radares e operações em pistas simples, priorizando noite e fim de semana | Ultrapassagem OR 1,6; velocidade OR 1,7; fim de semana +31% de letalidade |
| 3 | **Proteção ao pedestre** em travessias urbanas de rodovias: passarelas, faixas iluminadas, redutores e iluminação | Atropelamento: 16% das mortes, OR 6,6 |
| 4 | **Iluminação, sinalização refletiva e pontos de parada/descanso** para combater a fadiga; campanhas de sono ao volante | Madrugada 3× mais letal; sono/mal súbito com 11,8 mortos/100 |
| 5 | **Plano regional Norte/Nordeste** (MA, PA, TO, BA): investir em atendimento pré-hospitalar (tempo-resposta do SAMU) e em infraestrutura | Letalidade 1,8 a 2,5× a média nacional |
| 6 | **Monitoramento contínuo** com o painel, usando letalidade e mortes (não o total de acidentes) como indicadores oficiais | Total de acidentes distorcido por mudança de registro |

## 6. Conclusões

O risco de morrer nas rodovias federais está **concentrado e é previsível**: colisões frontais e atropelamentos, em pistas simples e rurais, à noite, em um conjunto identificável de trechos e estados. Isso é uma boa notícia para a política pública, porque permite **focalizar investimentos**. Duplicar ou separar sentidos nos trechos críticos e reforçar a fiscalização noturna de ultrapassagem e velocidade atacam diretamente quase metade das mortes. Sem essa mudança de foco, a tendência observada (mortes estáveis ou em alta desde 2020) torna inviável a meta de reduzir as mortes à metade até 2030.

**Limitações:**
- os dados cobrem apenas as rodovias **federais** policiadas pela PRF (não incluem estaduais nem municipais);
- as mortes são as registradas no local/ocorrência, e óbitos posteriores no hospital podem estar subnotificados;
- a mudança de taxonomia em 2021 exigiu agrupar as causas;
- falta um denominador de exposição (volume de tráfego/km rodado), então a letalidade compara gravidade, não risco por viagem.

## 7. Estrutura do repositório
```
├── README.md                         ← esta documentação
├── src/preparar_dados.py             ← limpeza, harmonização e agregações
├── notebooks/analise_exploratoria.ipynb ← EDA + regressão logística (executado)
├── dados/processados/                ← tabelas agregadas (CSV) usadas no painel
└── img/                              ← gráficos gerados no notebook
```

**Ferramentas:** Python (pandas, statsmodels, scikit-learn, matplotlib), Jupyter, Looker Studio, GitHub.

## Referências
- PRF — [Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf)
- CONTRAN — [PNATRANS (Plano Nacional de Redução de Mortes e Lesões no Trânsito)](https://www.gov.br/transportes/pt-br/assuntos/transito/conteudo-contran/resolucoes/Anexo_PNATRANS2023.pdf/@@download/file)
- Ministério dos Transportes — [OMS lança plano para reduzir acidentes e mortes no trânsito até 2030](https://www.gov.br/transportes/pt-br/assuntos/noticias/2021/10/oms-lanca-plano-para-reduzir-acidentes-e-mortes-no-transito-ate-2030)
- Portal do Trânsito — [Brasil tem até 2030 para reduzir mortes no trânsito pela metade](https://www.portaldotransito.com.br/noticias/mobilidade-e-tecnologia/seguranca/brasil-tem-ate-2030-para-reduzir-mortes-no-transito-pela-metade-numeros-mostram-o-tamanho-do-desafio/)
