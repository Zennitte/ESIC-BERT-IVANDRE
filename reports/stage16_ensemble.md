# Etapa 16 — Ensemble algorítmico

Família selecionada internamente: **16b_weighted**.
Pesos finais (BERT, TF-IDF, XGBoost): [0.6000000000000001, 0.3, 0.1].

| Modelo | Acurácia OOF | Macro F1 OOF |
|---|---:|---:|
| bert | 45.5707% | 45.3457% |
| tfidf | 45.0023% | 44.7351% |
| xgboost | 43.3033% | 42.6472% |
| 16a_equal | 46.3265% | 46.0589% |
| 16a_weighted | 45.8812% | 45.6356% |
| 16b_equal | 46.0394% | 45.7074% |
| 16b_weighted | 46.0218% | 45.7714% |

| Família | Acurácia interna média | Macro F1 interno médio |
|---|---:|---:|
| 16a_equal | 46.9510% | 46.6225% |
| 16a_weighted | 47.5889% | 47.3396% |
| 16b_equal | 46.4848% | 46.0961% |
| 16b_weighted | 47.6792% | 47.4167% |

Meta >=47% OOF atingida: **False**. Ganho vs BERT: 0.4511 pontos percentuais.

Pesos escolhidos em checkpoint_validation de cada fold; família escolhida pela média interna antes de consultar resultados externos. Grade de 10%; igualdade de acurácia desempatada por Macro F1. Empates exatos mantêm a primeira entrada.
TF-IDF: baseline C=0,5 congelado, vocabulário/IDF e classificador ajustados apenas no fit B. BERT e XGBoost reutilizam modelos de folds já congelados. A validação interna já participou da seleção anterior dos componentes; suas métricas são otimistas. Desenvolvimento reutilizado não constitui teste independente.
17.068 IDs OOF únicos; holdout histórico excluído da seleção. Final com 18.813 linhas B, incluindo holdout após seleção, sem sintéticos. Pesos finais são a média dos cinco vetores internos; OOF mede vetores por fold, não mede a versão final com pesos médios.
Pacote recarregável: `ensemble\models\best_ensemble`. Modelos anteriores, datasets e splits preservados.
Tempo desta execução: 336.1s. Etapa 17 aguarda autorização.

## Prioridade OOF — orientação posterior do usuário

Após a execução, o usuário orientou: “Então foque na oof”. A comparação principal passa a priorizar acurácia OOF, com Macro F1 OOF como desempate. A seleção interna original permanece registrada, incluindo o pacote triplo.

Melhor ensemble OOF observado: **BERT + TF-IDF 50/50 (16a_equal)**; acurácia **46.3265%**, Macro F1 **46.0589%**. Pacote final recarregável congelado em `ensemble/models/oof_best_dual_equal/`, com os componentes finais já treinados em 18.813 linhas B. Não houve novo treino, mudança de dados ou otimização adicional dos pesos. Meta >=47% OOF não atingida.

Essa escolha foi feita após consultar OOF reutilizada de desenvolvimento; é exploratória e precisa de avaliação independente para confirmação. Não é uma nova estimativa independente do pacote final.

## Análise pareada autorizada

Ganho vs TF-IDF: 1.3241 p.p. (226 acertos líquidos); IC95% bootstrap por componentes/fold [0.5737; 2.0524] p.p. O intervalo ficou inteiramente acima de zero: há suporte exploratório para ganho positivo, condicionado às previsões e grupos observados. Ver `stage16_paired_analysis.md`. Não é teste independente.
