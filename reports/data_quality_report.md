# Relatório final V2 — qualidade dos dados, experimentos e entrega

## Resultado e decisão

**Entregável escolhido: BERTimbau + TF-IDF/regressão logística, pesos 50/50.** Acurácia OOF de desenvolvimento **46,3265%**, Macro F1 **46,0589%**. A escolha foi feita após comparação na OOF, conforme orientação posterior do usuário; trata-se de resultado exploratório. A meta >=47% OOF não foi atingida.

**Conclusão frente à V1: inconclusiva em termos de superioridade confirmada.** O ensemble V2 tem valor numérico maior que a média CV V1 de 45,007% e o holdout histórico V1 de 45,966%, mas protocolos, número de folds e papel dos conjuntos diferem. O modelo final V1 também não tem acurácia conhecida em teste externo rotulado. Não é correto atribuir causalmente a diferença à limpeza, ao ensemble ou a vazamento na V1.

A comparação mais consistente disponível nesta rodada é ensemble vs TF-IDF nos mesmos 17.068 exemplos: ganho 1,3241 p.p., 226 acertos líquidos e vantagem nos cinco folds. A análise agrupada condicional sustenta um ganho exploratório; não constitui teste independente nem inclui variabilidade de novos treinos.

## 14 questões de síntese sobre dados e limites

| Nº | Questão | Evidência e conclusão |
|---|---|---|
| 1 | Qual o tamanho e o desbalanceamento? | 20.092 linhas: c1 6.347 (31,59%), c234 6.853 (34,11%), c5 6.892 (34,30%). Razão maior/menor 1,086; pequeno desbalanceamento global. |
| 2 | Há dados ausentes ou estruturalmente inválidos? | Zero textos/rótulos nulos ou vazios e zero labels fora das três classes. Uma célula de texto não string foi convertida com str(), conforme protocolo. |
| 3 | Há baixa informação ou truncamento? | 252 textos até cinco palavras; 2.174 (10,82%) acima de 512 tokens. Podem faltar contexto/anexos ou partes finais; efeito causal do truncamento não foi isolado. |
| 4 | Existem artefatos de coleta/encoding? | 5.919 linhas com controles Unicode C1, 7.936 com URLs e 2.175 com e-mails. Controles merecem inspeção da fonte; não houve correção forense nem comprovação de corrupção por linha. |
| 5 | Há duplicatas? | 18.432 textos exatos distintos; 463 grupos repetidos/2.123 linhas. Normalização para agrupamento: 18.143 chaves, 500 grupos repetidos/2.449 linhas. |
| 6 | Existem rótulos conflitantes? | 291 grupos exatos com 1.682 linhas; 316 grupos normalizados com 1.975 linhas. Contradição não determina qual rótulo é correto; contexto ausente ou subjetividade também são possíveis. |
| 7 | Há quase duplicatas e risco nas divisões? | 28.314 pares candidatos confirmados >=0,88; 3.044 >=0,99. V2 protege componentes exatos/normalizados e pares observados >=0,99, sem cruzamentos protegidos. Busca aproximada não exaustiva; não prova ausência de todo conteúdo semelhante. |
| 8 | Quantos rótulos parecem suspeitos e foram confirmados? | Triagem OOF de desenvolvimento: 4.347 alta prioridade e 3.685 média. Não são erros confirmados. Não há adjudicação independente nem rubrica disponível; zero relabeling. |
| 9 | Quais classes/fronteiras são mais difíceis? | c234 permanece a classe de menor F1. No ensemble atual, c234→c5 é a maior confusão dessa classe; métricas e matriz abaixo. Sobreposição semântica não foi adjudicada por humanos. |
| 10 | Que limpeza foi aplicada e quantos dados saíram? | B remove somente cópias extras de texto exato+mesmo label. Final: 1.279 removidas, 18.813 elegíveis; conflitos mantidos. C excluiu grupos exatos conflitantes só no treino, por autorização específica: 1.387 IDs únicos no desenvolvimento, sem alterar validações/rótulos. |
| 11 | A limpeza melhorou as métricas? | B: 44,9086% accuracy/44,6447% Macro F1; A: 44,2700%/43,6816%; C: 44,8500%/44,5803%. B teve melhor resultado agregado e menor intervenção; não houve ganho adicional de C sobre B. |
| 12 | Os sintéticos ajudaram e a proporção foi respeitada? | Etapa 12 reconciliou 3.918 novas aprovações sem duplicar candidate_id/fold; cotas F1–F5 cumpridas, dois excedentes F5 fora e não transferidos. Etapa 13 usou 90:10 e teve 44,2758% accuracy/44,4559% Macro F1, abaixo de B 100:0; c234 melhorou, c5 piorou. Sintéticos não foram usados nos modelos finais das Etapas 14–17. |
| 13 | O ganho do ensemble e sua estabilidade estão demonstrados? | Melhor observado: duplo 50/50, 46,3265% OOF, +1,3241 p.p. vs TF-IDF. Ganho em cinco folds e bootstrap agrupado positivo; seleção na OOF e dependência entre modelos limitam a conclusão. Variabilidade entre seeds não foi medida. |
| 14 | O limite vem dos dados, do modelo ou de ambos? | Há limitações observadas dos dados e do protocolo, mas não se identificou um teto de generalização. Dificuldade de c234, respostas sem contexto e truncamento são hipóteses; acurácia perto de 46% não comprova que o dataset impossibilite resultado maior. Falta confirmação independente e revisão de anotação. |

Fontes: `dataset_audit.md`, `duplicate_analysis.md`, `label_review.md`, `cleaning_policy.md`, `dataset_versions.md`, `split_protocol.md` e relatórios das Etapas 11–16. Seus SHA-256 estão em `data_quality_metrics.json`.

## Resultados OOF cumulativos

| Experimento | Acurácia OOF | Macro F1 OOF |
|---|---:|---:|
| 11/A original | 44.2700% | 43.6816% |
| 11/B limpeza básica | 44.9086% | 44.6447% |
| 11/C filtro de conflitos | 44.8500% | 44.5803% |
| 13/B + sintéticos 90:10 | 44.2758% | 44.4559% |
| 14a/BERTimbau softmax histórico | 43.9243% | 43.5803% |
| 14b/xgboost | 43.3033% | 42.6472% |
| 14d/bert-softmax | 43.8013% | 43.3319% |
| 14c/bert-ordinal | 42.8814% | 34.3466% |
| 15/BERT linear + warmup | 45.5707% | 45.3457% |
| 16/tfidf | 45.0023% | 44.7351% |
| 16/xgboost | 43.3033% | 42.6472% |
| 16/16a_equal | 46.3265% | 46.0589% |
| 16/16a_weighted | 45.8812% | 45.6356% |
| 16/16b_equal | 46.0394% | 45.7074% |
| 16/16b_weighted | 46.0218% | 45.7714% |

Os resultados acima são descritivos de desenvolvimento. Etapas iniciais selecionavam por Macro F1; posteriores por acurácia interna. A orientação de priorizar OOF veio depois da Etapa 16. Seleções e checkpoints anteriores foram preservados. TF-IDF da Etapa 7 usou todo o treino externo e obteve 45,0785%; a versão comparável da Etapa 16 usa fit B e obteve 45,0023%, portanto esses baselines não devem ser confundidos.

## Ensemble entregue: desempenho por classe e confusão

| Classe | F1 OOF |
|---|---:|
| c1 | 45.8738% |
| c234 | 39.8251% |
| c5 | 52.4780% |

Matriz de confusão: linhas são classes reais; colunas são predições.

| Real / Predita | c1 | c234 | c5 |
|---|---:|---:|---:|
| c1 | 2268 | 1666 | 1461 |
| c234 | 1406 | 2277 | 2134 |
| c5 | 819 | 1675 | 3362 |

## Acertos compartilhados e incerteza

Ensemble e TF-IDF acertam juntos 6.403 textos e erram juntos 7.883; só o ensemble acerta 1.504 e só o TF-IDF acerta 1.278. São 226 acertos líquidos a favor do ensemble.

20.000 reamostragens pareadas de 14.712 componentes protegidos, estratificadas por fold: IC95% do ganho +0,5737 a +2,0524 p.p. McNemar exato por linha p=0,00001972 é apenas sensibilidade descritiva, pois a independência das linhas não está garantida. A reamostragem não retreina e não corrige o efeito da seleção na OOF. Ver `stage16_paired_analysis.md`.

## Consistência dos rótulos e teto: o que pode ser calculado

Na tabela original, atribuir uma única classe determinística a cada texto exatamente igual permite no máximo 19307/20.092 acertos (96.0930%), escolhendo a classe majoritária por texto. Se o classificador tratasse caixa/NFC/espaços como iguais, o limite descritivo seria 95.4260%.

Esses limites apenas quantificam contradições na tabela observada. Não são acurácia atingível em novos dados, não medem capacidade de generalização e não explicam sozinhos o patamar de 46%. O modelo recebe texto original, portanto a versão normalizada é um cenário de invariância, não uma restrição aplicada à entrada. Não é possível separar quantitativamente limite dos dados e limite do modelo com os experimentos atuais.

## Entregável e inferência final

`v2/deliverable/` é autônomo: model.py, predict.py, train.py opcional, requirements.txt, README, config, artefatos e tokenizer, dados copiados/exportados, relatórios e manifests.

BERT e TF-IDF finais foram treinados em 18.813 linhas B originais; BERT veio da Etapa 15 (3.557 updates, LR 2e-5, linear+warmup 10%, seed 20260941) e TF-IDF da Etapa 16. A Etapa 17 copiou os pesos sem treinar ou converter novamente. O holdout histórico foi incluído nesses ajustes finais; não é um teste independente.

Predição de **900 textos** de test1.xlsx concluída em **40.83s**, dispositivo **cuda**. Distribuição predita: {'c234': 300, 'c5': 324, 'c1': 276}. Arquivo: `deliverable/predictions/test_predito.xlsx`.

Entrada SHA-256 `e626f030d4bf2be889fc41dd1fe9b81fb31f1f50f3ace50530d658683260449f`; saída `4c34dfd617d2ac8e493f6e44d40574fd27bdf874f290570ca152e70efb31d603`. Foram conferidos 1802 valores de células; somente clarity nas 900 linhas previstas mudou, todas nas três classes válidas. Textos/entrada original preservados.

O teste não tem rótulos: **acurácia e Macro F1 de teste desconhecidos**. A planilha contém predições, não gabarito nem confirmação da meta de 47%. Inferência padrão CPU, opção GPU. Reconstrução opcional não executada e não alegada como verificada nesta etapa.

## Limitações e decisão final

- Não há teste externo rotulado independente para o pacote final.
- A OOF foi reutilizada em várias decisões; a seleção atual pelo melhor resultado OOF é exploratória.
- Uma seed por fold; não houve repetição de treinamento entre seeds.
- Folds e grupos compartilhados limitam inferência estatística simples por linha.
- Triagem de ruído não foi validada por adjudicação humana; nenhum rótulo foi substituído por previsão.
- Busca de quase duplicatas não exaustiva; similaridade de modelos de resposta pode gerar falsos positivos.
- Controles de encoding e truncamento não tiveram ablações isoladas.
- Sintéticos 90:10 não melhoraram as métricas globais neste protocolo; não generalizar para toda augmentation.

A V2 produziu uma auditoria rastreável, intervenções comparáveis e um ensemble com ganho OOF observado frente ao TF-IDF da mesma rodada. O teto do dataset e superioridade sobre a V1 continuam não confirmados. Entregar o duplo 50/50, com as limitações explícitas, e preservar todos os resultados anteriores.

## Proveniência

Detalhes estruturados, métricas exatas e hashes de fontes: `data_quality_metrics.json`. Inferência: `deliverable/predictions/test_predito.metadata.json`. Pacote completo: `deliverable/package_manifest.json`. Execução/autorização/progresso: `experiments/stage17/` e `plan.md`.
