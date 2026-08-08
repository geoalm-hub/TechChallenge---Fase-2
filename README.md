# Wine Quality Classification

Tech Challenge — Fase 2 | Pós-Graduação em Data Analytics (FIAP)

Classificação binária da qualidade de vinhos com base em características físico-químicas, utilizando o [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset).

## 1. Contexto e Objetivo

A avaliação da qualidade de um vinho é tradicionalmente feita por especialistas, por meio de análise sensorial, um processo subjetivo e dependente da experiência do avaliador. Este projeto explora se características **físico-químicas** (acidez, teor alcoólico, densidade, dióxido de enxofre, entre outras) são suficientes para prever a qualidade de um vinho de forma automatizada, apoiando a tomada de decisão de enólogos e produtores.

**Definição da variável alvo:** a nota original de qualidade (`quality`, de 3 a 8) foi transformada em uma classificação binária:
- **Alta qualidade:** nota ≥ 7
- **Baixa/média qualidade:** nota < 7

## 2. Sobre o Dataset

- **Fonte:** [Wine Quality Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- **1.143 amostras**, 11 variáveis físico-químicas + `quality` (nota original) + `Id`
- Sem valores nulos ou duplicados

## 3. Análise Exploratória (EDA)

- **Desbalanceamento de classes:** 86% dos vinhos são de baixa/média qualidade e apenas 14% de alta qualidade — desbalanceamento relevante, que orientou a escolha das métricas de avaliação (acurácia sozinha seria enganosa) e motivou o uso de SMOTE no pré-processamento.
- **Correlação com a variável alvo (Pearson):** `alcohol` (0.48) e `volatile acidity` (-0.41) apresentaram as correlações mais fortes com a qualidade.
- **Outliers:** identificados via boxplot em praticamente todas as variáveis (destaque para `residual sugar`, `sulphates` e `volatile acidity`). Ao comparar as distribuições segmentadas por classe (alta vs. baixa/média qualidade), observou-se que os outliers seguem um padrão coerente com a qualidade (ex.: outliers de alta acidez volátil concentram-se em vinhos de baixa/média qualidade), evidência de que representam variação real do processo produtivo, e não erros de medição. Por isso, optou-se por **mantê-los** no dataset.
  - O dataset não possui nenhuma variável categórica (método de produção, região, vinícola), o que limita uma investigação mais aprofundada da causa dos outliers.
- **Multicolinearidade:** identificada entre `fixed acidity`, `pH` e `density` (correlações de até -0.69), esperada quimicamente, já que pH é uma medida direta de acidez. Não compromete os modelos utilizados (KNN e Árvore de Decisão são relativamente robustos a essa questão).

## 4. Pré-processamento

1. **Divisão treino/teste:** `train_test_split` com `test_size=0.25` (75% treino / 25% teste) e `stratify=y`, garantindo que a proporção 86%/14% se mantivesse em ambos os conjuntos.
2. **Padronização:** `StandardScaler`, ajustado (`fit`) apenas no conjunto de treino e aplicado (`transform`) no treino e teste, evitando vazamento de dados (*data leakage*).
3. **Balanceamento de classes:** aplicado **SMOTE** (Synthetic Minority Oversampling Technique) exclusivamente no conjunto de treino já padronizado, equilibrando as classes em 738 amostras cada. O conjunto de teste foi mantido com a proporção original (86%/14%), representando a realidade dos dados.

## 5. Modelos Desenvolvidos

Foram treinados e comparados **dois algoritmos de classificação**, com lógicas de decisão distintas:

- **KNN (K-Nearest Neighbors):** classifica um vinho com base no "voto" dos K vizinhos mais próximos (sensível à escala das variáveis, por isso a padronização foi essencial). Testado com `k=6` e, posteriormente, `k=3`.
- **Árvore de Decisão:** classifica por meio de uma sequência de divisões binárias sobre as features (ex.: "alcohol > 11%?").

Cada modelo foi treinado com e sem balanceamento via SMOTE, totalizando 4 configurações comparadas.

> Optou-se por não incluir Regressão Logística como terceiro modelo, mantendo o escopo em KNN e Árvore de Decisão (mínimo exigido pelo desafio), focando o esforço na comparação, interpretação e no tratamento do desbalanceamento de classes.

## 6. Avaliação dos Modelos

Dado o desbalanceamento das classes, a **acurácia isolada não é uma métrica confiável** (um modelo que sempre prevê "baixa/média qualidade" já atingiria ~86% de acurácia sem aprender nada). A avaliação priorizou **precision, recall e f1-score da classe "alta qualidade"**, com foco especial em **precision**, sob a ótica de que classificar um vinho ruim como bom representa um risco maior ao negócio do que o inverso.

| Modelo | Precision | Recall | F1-score | Accuracy |
|---|---|---|---|---|
| KNN (k=6) | 0.528 | 0.475 | 0.500 | 0.867 |
| KNN (k=3) + SMOTE | 0.408 | 0.725 | 0.523 | 0.815 |
| Árvore de Decisão | 0.500 | 0.500 | 0.500 | 0.860 |
| **Árvore de Decisão + SMOTE** | **0.500** | **0.650** | **0.565** | **0.860** |

*(métricas referentes à classe "Vinho de alta qualidade")*

**Modelo escolhido: Árvore de Decisão + SMOTE.** Manteve a mesma precision da árvore original (não piorou a confiabilidade das previsões positivas) e apresentou o melhor recall e f1-score entre as quatro configurações — o melhor equilíbrio frente ao critério de negócio definido.

## 7. Interpretação dos Resultados

Importância das variáveis extraída da Árvore de Decisão (`feature_importances_`):

| Variável | Importância |
|---|---|
| alcohol | 41.3% |
| sulphates | 11.3% |
| volatile acidity | 11.3% |
| fixed acidity | 6.6% |
| citric acid | 6.4% |
| total sulfur dioxide | 5.7% |
| chlorides | 5.0% |
| free sulfur dioxide | 4.3% |
| residual sugar | 3.6% |
| density | 3.0% |
| pH | 1.4% |

**Principais achados e implicações para o processo produtivo:**

- **Teor alcoólico (alcohol)** é, disparado, o fator mais determinante para a classificação de alta qualidade — vinhos com maior teor alcoólico tendem a receber notas mais altas. Como o álcool resulta da conversão de açúcar durante a fermentação, esse é um ponto que deve ser monitorado e controlado de perto durante esse processo.
- **Acidez volátil (volatile acidity)** apresenta relação inversa: quanto menor, maiores as chances de o vinho ser classificado como alta qualidade. Acidez volátil elevada costuma estar associada à contaminação bacteriana ou oxidação, reforçando a importância de rigor na higiene do processo produtivo e de evitar exposição do produto ao ar durante a fabricação.
- **Sulfatos (sulphates)** tiveram correlação linear discreta (Pearson = 0.26), mas ganharam mais peso na árvore (11.3%), sugerindo uma relação não-linear com a qualidade, melhor capturada pelo modelo do que pela correlação simples. Sulfatos estão ligados à conservação e ao controle microbiológico do vinho, reforçando a importância de seu monitoramento mesmo quando seu efeito isolado parece pequeno.

**Limitações reconhecidas:**
- A qualidade do vinho é, em sua origem, uma nota sensorial subjetiva atribuída por especialistas, nem toda a variação da nota é explicada pelas variáveis físico-químicas disponíveis, o que impõe um teto natural ao desempenho de qualquer modelo treinado apenas com esses dados.
- A classe "alta qualidade" possui poucas amostras no dataset original (159 de 1.143), o que limita a capacidade de generalização do modelo mesmo após o balanceamento com SMOTE.

## 8. Estrutura do Repositório

```
wine-quality-classification/
│
├── data/              # Base de dados utilizada (WineQT.csv)
├── notebooks/         # Notebook com a análise e modelagem
├── src/                # Scripts auxiliares (pré-processamento ou modelagem)
├── results/            # Gráficos e métricas dos modelos
├── requirements.txt    # Bibliotecas utilizadas
└── README.md           # Este arquivo
```

## 9. Como Executar

```bash
git clone https://github.com/geoalm-hub/TechChallenge---Fase-2
cd wine-quality-classification
pip install -r requirements.txt
```

Abra o notebook em `notebooks/` e execute as células em sequência.

## 10. Tecnologias Utilizadas

- Python 3
- pandas, numpy
- scikit-learn (StandardScaler, train_test_split, KNeighborsClassifier, DecisionTreeClassifier, classification_report)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

## Autor(es)

- Geovana de Almeida Nunes
