# Análise pareada OOF — ensemble duplo versus TF-IDF

Base: 17,068 exemplos, cinco folds, 14,712 componentes protegidos. Não houve novos treinos.

| Resultado pareado | Textos |
|---|---:|
| Ambos acertam | 6403 |
| Ambos erram | 7883 |
| Só ensemble acerta | 1504 |
| Só TF-IDF acerta | 1278 |

Saldo: **226 acertos adicionais**; ensemble **46.3265%**, TF-IDF **45.0023%**, ganho **1.3241 p.p.**.

## Incerteza agrupada — análise principal
Bootstrap pareado por componentes protegidos, estratificado por fold; 20,000 reamostragens, seed 20261002. Intervalo percentil de 95% para o ganho: **[0.5737; 2.0524] p.p.**.
O intervalo ficou inteiramente acima de zero: há suporte exploratório para ganho positivo, condicionado às previsões e grupos observados.
Os componentes são os mesmos do split congelado: texto normalizado exato e conexões observadas de quase duplicatas com similaridade >=0,99, com fechamento transitivo. Cada componente é mantido inteiro na reamostragem; contagem de componentes por fold fixa e número de linhas variável. O resultado agregado continua ponderado por texto.

## McNemar por linha — sensibilidade descritiva
Teste binomial exato bilateral sobre 2782 pares discordantes: p=1.97201e-05.
Esse teste pressupõe pares de linhas independentes; duplicatas e modelos compartilhados por fold enfraquecem essa hipótese. Não interpretar seu p-valor como confirmação independente; a análise agrupada é a referência principal.

| Fold | TF-IDF | Ensemble | Ganho (p.p.) |
|---|---:|---:|---:|
| 1 | 45.5891% | 46.8369% | +1.2478 |
| 2 | 44.8110% | 46.4007% | +1.5897 |
| 3 | 44.2977% | 45.6783% | +1.3806 |
| 4 | 45.9841% | 47.5728% | +1.5887 |
| 5 | 44.3351% | 45.1785% | +0.8434 |

## Limites
Análise exploratória posterior à escolha do ensemble pela OOF reutilizada. O intervalo é condicional às previsões salvas, não incorpora incerteza de novos treinos/seeds nem corrige o efeito da seleção entre modelos. Bootstrap não transforma desenvolvimento em teste independente. O desempenho do pacote final em novos dados permanece desconhecido.
Dados, rótulos, splits e modelos congelados não foram modificados. Etapa 17 não foi iniciada.

## Artefatos
`ensemble/diagnostics/paired_analysis/result.json`, `paired_predictions.csv`, `bootstrap_replicates.npz`; script `ensemble/src/analyze_paired_gain.py`.
Referências metodológicas: [McNemar](https://www.statsmodels.org/v0.14.4/generated/statsmodels.stats.contingency_tables.mcnemar.html), [bootstrap pareado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
