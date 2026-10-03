# ESIC-BERT-IVANDRE — classificação de clareza e-SIC (V2)

Entrega autônoma para classificar textos de respostas e-SIC nas classes **c1, c234 e c5**, combinando **BERTimbau + TF-IDF/regressão logística com pesos 50/50**. Este repositório publica o conteúdo de `v2/deliverable/` na raiz: código, modelos treinados, tokenizer, dados, previsões e relatórios.

A inferência funciona em CPU com arquivos locais, sem retreinar e sem baixar o encoder base. A reconstrução do treino é opcional e exige GPU.

## Contexto geral

A V2 investigou qualidade dos dados, duplicatas, conflitos de rótulos, políticas de limpeza e diferentes classificadores. O pacote escolhido usa a **política B**: remover cópias redundantes de texto exatamente igual com mesmo rótulo, mantendo conflitos. Os componentes finais foram ajustados em **18.813 exemplos originais**, incluindo o holdout histórico após a seleção, sem dados sintéticos.

Os nomes preservam a codificação da fonte: `c1` corresponde à categoria 1, `c234` agrupa as categorias 2, 3 e 4, e `c5` corresponde à categoria 5. A entrega não inclui a rubrica de anotação para definir operacionalmente cada nível; consulte a anotação de origem para interpretar esses níveis de clareza.

O ensemble duplo foi escolhido depois de comparar a OOF (*out-of-fold*: cada exemplo de desenvolvimento previsto por um modelo de fold que não o utilizou em seu ajuste). A escolha é exploratória. Este pacote contém a entrega final, não todo o ambiente dos experimentos anteriores. Referências históricas nos relatórios podem apontar para scripts e arquivos que não estão neste repositório.

## Instalação

### Clonar e obter os pesos

Instale Git e Git LFS. Os pesos BERT têm aproximadamente **436 MB** e estão em Git LFS:

```bash
git lfs install
git clone https://github.com/Zennitte/ESIC-BERT-IVANDRE.git
cd ESIC-BERT-IVANDRE
git lfs pull
```

O ZIP do GitHub pode conter apenas o ponteiro LFS, dependendo da configuração do repositório. O procedimento recomendado para obter o conteúdo real é clonar e executar `git lfs pull`.

### Ambiente Python

Use uma versão Python compatível com as dependências de `requirements.txt`. As versões registradas no ambiente de origem estão em [provenance/runtime_versions.json](provenance/runtime_versions.json); elas documentam aquela execução, sem exigir o mesmo build de GPU.

Windows/PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se a ativação for bloqueada no PowerShell, use diretamente `.\.venv\Scripts\python.exe` para instalar dependências e executar os scripts.

CPU é o padrão de inferência. Para GPU, instale/configure PyTorch com suporte CUDA ou ROCm compatível com o hardware. As dependências Python, sozinhas, não configuram esse suporte. O código usa a primeira GPU como `cuda:0`, inclusive no ambiente ROCm. Não há suporte específico a MPS neste pacote. O BERT inteiro é carregado em memória; reduza o lote em caso de falta de memória na GPU.

## Como prever em uma planilha

### Teste incluído

Execute na raiz do repositório:

```bash
python predict.py --input data/test.xlsx --sheet test1 --output predictions/minha_execucao.xlsx --device cpu --batch-size 4
```

O nome novo inicia uma inferência completa. `python predict.py` também funciona, usando os caminhos padrão e a saída histórica `predictions/test_predito.xlsx`.

GPU configurada:

```bash
python predict.py --input data/test.xlsx --sheet test1 --output predictions/minha_execucao_gpu.xlsx --device cuda --batch-size 4
```

### Seus próprios textos

Prepare um arquivo **XLSX**, com cabeçalhos na primeira linha:

| resp_text | clarity |
|---|---|
| Texto da primeira resposta | célula vazia |
| Texto da segunda resposta | célula vazia |

- Os cabeçalhos são `resp_text` e `clarity`; espaços nas bordas dos cabeçalhos são removidos.
- A coluna `clarity` deve estar vazia nas linhas com texto. Uma entrada já rotulada é recusada.
- Linhas com `resp_text` igual a `None` são ignoradas. Outros valores não textuais são convertidos por `str()`.
- Envie texto original, sem limpeza adicional. Use valores de texto nas células, não fórmulas que produzam o texto.
- Entrada e saída devem ser arquivos distintos.
- Podem existir outras colunas e abas. O script verifica os valores de todas as células e admite mudança apenas em `clarity` nas linhas previstas. Isso não garante preservação de todos os recursos avançados de Excel, como objetos e formatação especial.

```bash
python predict.py --input meus_dados.xlsx --sheet Respostas --output predictions/meus_dados_preditos.xlsx
```

| Parâmetro | Padrão | Função |
|---|---|---|
| `--input` | `data/test.xlsx` junto do script | Entrada sem rótulos |
| `--output` | `predictions/test_predito.xlsx` junto do script | Planilha preenchida |
| `--sheet` | `test1` | Aba de entrada |
| `--batch-size` | `4` | Textos por lote; inteiro positivo |
| `--device` | `cpu` | `cpu` ou `cuda` |
| `--artifacts` | `artifacts/` junto do script | Pasta do ensemble |

Caminhos padrão usam a localização do script. Caminhos relativos informados como argumentos usam o diretório atual. Consulte `python predict.py --help`.

### Saídas, interrupção e retomada

Para a saída `minha_execucao.xlsx`, são gerados:

| Arquivo | Conteúdo |
|---|---|
| `minha_execucao.xlsx` | Dados originais com `clarity` preenchida |
| `minha_execucao.checkpoint.json` | Linhas, rótulos, probabilidades na ordem c1/c234/c5 e hashes para retomada |
| `minha_execucao.metadata.json` | Contagens por classe, duração, dispositivo, hashes e verificações |

O checkpoint é salvo a cada bloco de até 64 textos, independentemente do lote. Para retomar uma interrupção, repita o mesmo comando. Entrada, aba, linhas e manifesto de artefatos devem coincidir com o checkpoint. Um checkpoint completo permite regravar a saída sem recalcular o modelo; use um nome de saída novo para recalcular desde o início.

Para pausar, crie `PAUSE_REQUESTED` na raiz do pacote: `New-Item PAUSE_REQUESTED -ItemType File` no PowerShell ou `touch PAUSE_REQUESTED` no Linux/macOS. A pausa ocorre após salvar um bloco e termina com código **75**. Remova a marca e execute novamente para retomar.

## Como usar o modelo em Python

Execute a partir da raiz, com o ambiente ativado:

```python
from model import Ensemble

modelo = Ensemble(device="cpu")  # verifica hashes e carrega modelos locais
textos = [
    "A informação solicitada está disponível no documento anexado.",
    "O pedido foi encaminhado ao setor responsável para análise.",
]
probabilidades = modelo.predict_proba(textos, batch_size=4)
classes = modelo.cfg["class_order"]  # ["c1", "c234", "c5"]
rotulos = [classes[i] for i in probabilidades.argmax(axis=1)]
for texto, rotulo, probs in zip(textos, rotulos, probabilidades):
    print(texto, rotulo, dict(zip(classes, probs.tolist())))

# Para obter apenas rótulos em outra chamada:
# rotulos = modelo.predict(textos, batch_size=4)
```

`predict_proba` retorna um array `(n_textos, 3)`; `predict` retorna uma lista de rótulos. Ambas executam inferência: aproveite as probabilidades já obtidas se precisar dos dois resultados. Reutilize a instância de `Ensemble` entre chamadas. Uma lista vazia retorna resultados vazios.

As probabilidades são as saídas combinadas dos classificadores. A entrega não realizou estudo de calibração que permita interpretá-las como confiança validada.

## Modelo e processamento

```text
p_final = 0,5 × softmax(logits_BERT) + 0,5 × predict_proba_TFIDF
rótulo = class_order[argmax(p_final)]
class_order = [c1, c234, c5]
```

As colunas do TF-IDF são reordenadas para coincidir com as do BERT. A média é feita sobre probabilidades, não sobre rótulos.

| Componente | Configuração |
|---|---|
| Encoder base | `neuralmind/bert-base-portuguese-cased` |
| Revisão base | `94d69c95f98f7d5b2a8700c420230ae10def0baa` |
| BERT final | Classificação de sequência com três saídas, pesos safetensors |
| Tokenização BERT | Texto preservado, início de até 512 tokens incluindo especiais, truncamento e padding dinâmico |
| TF-IDF | Palavras, unigramas/bigramas, min_df=3, max_features=100000, sublinear_tf=True, lowercase=True, norma L2 |
| Regressão logística | C=0.5, solver=lbfgs, max_iter=400, random_state=20260924 |
| Pesos | BERT 0.5; TF-IDF 0.5 |

O TF-IDF recebe o texto original e aplica suas transformações no pipeline serializado. BERT/tokenizer carregam com `local_files_only=True`. XGBoost foi investigado no desenvolvimento, mas não integra o pacote selecionado.

## Dados e proveniência

| Conjunto | Arquivo / aba | Linhas de dados | Uso |
|---|---|---:|---|
| Treino final | `data/train.xlsx` / `train` | 18.813 | Ajuste dos componentes, colunas resp_text/clarity |
| Teste sem gabarito | `data/test.xlsx` / `test1` | 900 | Inferência, clarity originalmente vazia |
| Teste predito | `predictions/test_predito.xlsx` / `test1` | 900 | Previsões históricas da entrega |

A fonte tinha 20.092 linhas. A política B retirou 1.279 cópias redundantes de texto exato+mesmo rótulo, mantendo conflitos sem reanotação. O arquivo original da V1 não está incluído; seu hash consta em `config.json`. A correspondência com IDs da fonte está em `provenance/training_ids.json`.

Distribuição verificada no treino final: **5.753 c1, 6.462 c234 e 6.598 c5**.

Os **17.068 IDs OOF** pertencem ao desenvolvimento; o treino final tem 18.813 linhas. As divisões históricas protegeram componentes de textos exatos/normalizados e conexões observadas de quase duplicatas com similaridade ≥0,99. A busca foi aproximada e não exaustiva.

O teste predito histórico contém **276 c1, 300 c234 e 324 c5**. São previsões, não gabarito. O pacote não inclui uma licença própria para código/dados ou documentação de direitos de redistribuição da fonte; esta publicação não atribui uma licença adicional.

## Avaliação e limites

Nos mesmos 17.068 IDs OOF de desenvolvimento:

| Modelo | Acurácia OOF | Macro F1 OOF |
|---|---:|---:|
| BERT | 45,5707% | 45,3457% |
| TF-IDF + regressão logística | 45,0023% | 44,7351% |
| **Ensemble 50/50** | **46,3265%** | **46,0589%** |

| Classe | F1 OOF |
|---|---:|
| c1 | 45,8738% |
| c234 | 39,8251% |
| c5 | 52,4780% |

Matriz de confusão (linhas reais; colunas previstas):

| Real / prevista | c1 | c234 | c5 |
|---|---:|---:|---:|
| c1 | 2268 | 1666 | 1461 |
| c234 | 1406 | 2277 | 2134 |
| c5 | 819 | 1675 | 3362 |

Ganho frente ao TF-IDF: **1,3241 p.p.**, 226 acertos líquidos. Bootstrap pareado agrupado por componentes/fold: IC95% condicional **+0,5737 a +2,0524 p.p.**; ver [análise pareada](reports/stage16_paired_analysis.md).

A seleção consultou OOF reutilizada: o resultado é exploratório, sem confirmação independente ou repetição entre seeds. Essas métricas não avaliam o modelo final sobre novos dados independentes. **A acurácia nos 900 textos de teste é desconhecida**, pois não há gabarito. A meta histórica de 47% OOF não foi atingida; a superioridade frente à V1 permanece inconclusiva.

Conflitos de rótulos, contexto ausente e truncamento são limitações documentadas. c234 tem o menor F1. Não foi estabelecido teto de generalização ou isolado efeito causal de cada intervenção. Veja [relatório de qualidade](reports/data_quality_report.md).

## Como realizar o treinamento

### Reconstrução opcional do procedimento final

`train.py` reconstrói o ajuste final de BERT e TF-IDF nos dados congelados. Não é necessário para usar a entrega. Exige GPU CUDA/ROCm disponível ao PyTorch, encoder base na revisão fixada acessível pela rede ou cache Hugging Face, dependências instaladas, dados originais da entrega e diretório de saída vazio/inexistente.

```bash
python -c "import torch; print(torch.cuda.is_available())"
python train.py
```

Ou escolha outro diretório:

```bash
python train.py --output minha_reconstrucao
```

| Parâmetro BERT | Valor |
|---|---|
| Seed | 20260941 |
| Otimizador | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Microbatch | 2 textos |
| Acumulação | 4 microbatches, 8 textos por atualização completa |
| Gradient clipping | Norma máxima 1 |
| Scheduler | Linear, horizonte de 4.702 updates |
| Warmup | 470 updates, aproximadamente 10% do horizonte |
| Parada | Fixa em 3.557 updates; sem early stopping |
| Updates por época | 2.351 |
| Comprimento | Até 512 tokens |

O script verifica hash, cabeçalho e quantidade de linhas do treino; embaralha com seeds determinadas por época; ajusta BERT e depois TF-IDF/regressão logística nas mesmas linhas. Salva pesos, tokenizer, TF-IDF, pipeline e manifesto. A saída padrão é `retrained_artifacts/`; o script recusa sobrescrever `artifacts/` ou qualquer saída não vazia. Não oferece retomada por checkpoint de treinamento.

```bash
python predict.py --artifacts retrained_artifacts --output predictions/test_retreinado.xlsx --device cuda
```

O script opcional não foi executado para produzir a entrega histórica da Etapa 17, que reutilizou componentes congelados. Não foi executado novamente nesta publicação. Resultados bit a bit não são garantidos entre ambientes/hardware.

### Outros dados e reprodução da OOF

O script é específico para os dados congelados: não aceita `--input`, verifica hash e contagem e mantém alguns valores de lote/comprimento/horizonte no código. Alterar somente a planilha causa erro. Para novos dados, adapte leitura, configuração, validações e cronograma em uma cópia do projeto, preservando a entrega como referência.

Este repositório não contém os cinco modelos de folds, todos os scripts de seleção, todas as divisões ou previsões OOF. **`train.py` ajusta os componentes finais; não refaz a validação cruzada nem calcula as métricas OOF.** Em uma nova avaliação, separe dados rotulados independentes antes da seleção e mantenha duplicatas relacionadas como grupos nas divisões. Avaliar nas próprias linhas de treino não mede generalização.

## Tabela de arquivos

| Arquivo | Para que serve |
|---|---|
| `README.md` | Contexto, instalação, uso, treino, resultados e inventário |
| `requirements.txt` | Dependências Python e versões de serialização |
| `model.py` | Ensemble, carregamento local, hashes e previsão de probabilidades/rótulos |
| `predict.py` | CLI XLSX, checkpoint, retomada e verificação de células |
| `train.py` | Reconstrução opcional dos componentes finais |
| `config.json` | Classes, parâmetros, hashes dos dados e métricas |
| `.gitattributes` | Git LFS dos pesos e preservação dos bytes do pacote |
| `.gitignore` | Exclui ambientes, caches, checkpoints e retreino local |
| `package_manifest.json` | SHA-256 dos arquivos publicados, exceto o próprio manifesto |
| `artifacts/manifest.json` | Hashes e identificação dos componentes congelados |
| `artifacts/pipeline.json` | Ordem das classes/componentes, pesos e inferência |
| `artifacts/tfidf.joblib` | Pipeline treinado TF-IDF/regressão logística |
| `artifacts/bert/model.safetensors` | Pesos BERT ajustados, via Git LFS |
| `artifacts/bert/config.json` | Arquitetura e mapeamento de classes |
| `artifacts/bert/tokenizer.json` | Tokenizer serializado |
| `artifacts/bert/tokenizer_config.json` | Configuração da tokenização |
| `artifacts/bert/special_tokens_map.json` | Tokens especiais |
| `artifacts/bert/vocab.txt` | Vocabulário |
| `data/train.xlsx` | 18.813 exemplos elegíveis, aba train |
| `data/test.xlsx` | 900 exemplos sem rótulo, aba test1 |
| `predictions/test_predito.xlsx` | Previsão histórica do teste |
| `predictions/test_predito.metadata.json` | Auditoria histórica com hashes e caminhos de origem |
| `provenance/training_ids.json` | Vínculo com IDs da fonte |
| `provenance/runtime_versions.json` | Versões do ambiente histórico |
| `provenance/oof_selection.json` | Registro da seleção posterior pela OOF |
| `provenance/evaluation_metrics.json` | Métricas históricas detalhadas |
| `provenance/publication.json` | Manifesto original e alterações desta publicação |
| `reports/data_quality_report.md` | Qualidade, experimentos e limites |
| `reports/data_quality_metrics.json` | Evidências quantitativas e referências de auditoria |
| `reports/stage16_ensemble.md` | Seleção interna inicial e escolha OOF posterior |
| `reports/stage16_paired_analysis.md` | Comparação pareada e incerteza agrupada |

Checkpoints são gerados localmente e ignorados pelo Git. O checkpoint da execução histórica não foi publicado; planilha predita e metadados foram preservados.

## Integridade e solução de problemas

`Ensemble` verifica hashes dos componentes ao carregar. Para conferir a publicação inteira, depois de `git lfs pull`:

```python
import json
from pathlib import Path
from model import digest

raiz = Path(".")
manifesto = json.loads((raiz / "package_manifest.json").read_text(encoding="utf-8"))
for nome, esperado in manifesto["files_sha256"].items():
    assert digest(raiz / nome) == esperado, f"Hash diferente: {nome}"
print("Todos os arquivos conferidos.")
```

| Sintoma | Ação |
|---|---|
| Artifact hash differs | Execute git lfs pull; confira pesos reais e arquivos alterados |
| GPU unavailable | Use --device cpu ou configure PyTorch compatível com GPU |
| Memória GPU insuficiente | Reduza --batch-size, começando por 1 |
| Expected resp_text and clarity columns | Confira primeira linha e nome da aba |
| Input clarity must be empty | Use entrada sem rótulos |
| Resume input changed | Escolha novo nome de saída para novos dados/artefatos |
| Código 75, sem saída final | Remova PAUSE_REQUESTED e repita o comando |
| Training data changed | Restaure os dados congelados; novos dados exigem adaptação |
| ModuleNotFoundError | Use o Python do ambiente com requirements instalado |

Os manifestos publicados usam `/` nos caminhos para compatibilidade Windows/Linux. Pesos, dados e previsões históricas mantêm hashes originais. A normalização mudou o hash do JSON de artefatos; o hash antigo nos metadados históricos continua identificando o manifesto daquela execução. [provenance/publication.json](provenance/publication.json) preserva o registro original e documenta essa correspondência.
