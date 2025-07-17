## Documentação do Projeto: IA de Xadrez (AlphaZero Style)

### 1. Visão Geral do Projeto

**Objetivo**

* Desenvolver uma IA de xadrez inspirada no AlphaZero, com duas fases principais: treinamento supervisionado e autojogo (self‑play).

**Escopo**

* Geração de posições via `python-chess` e obtenção de gabarito (movimentos ideais) usando Stockfish.
* Pré‑processamento e formatação das posições de treino.
* Implementação de arquitetura neural para previsão de movimentos.
* Implementação de MCTS para geração de jogos por self‑play.
* Avaliação contínua contra Stockfish.

**Tecnologias**

* Linguagem: Python 3.x
* Bibliotecas: `python-chess`, Stockfish
* Framework ML: PyTorch (ou TensorFlow)
* Gerenciamento de experimentos e logs: MLflow / TensorBoard

---

### 2. Arquitetura e Fluxo de Dados

1. **Coleta de Dados**

   * Geração de posições aleatórias ou especificadas via `python-chess`.
   * Avaliação de cada posição pelo Stockfish para obter o movimento ótimo (gabarito).
   * Parâmetros iniciais: `stockfish_depth = 12`, `positions_to_generate = 1000`.

2. **Pré‑processamento**

   * Converter cada posição em tensor de features adequado (tabuleiro, histórico de jogadas, etc.).
   * Codificar o movimento-rotulo a partir do gabarito do Stockfish.

3. **Dataset de Treino**

   * Divisão em conjuntos de treino e validação.
   * Armazenamento em formato eficiente (por exemplo, HDF5 ou TorchDataset).

---

### 3. Treinamento Supervisionado

#### 3.1 Geração de Dados Iniciais

* **Fluxo**:

  1. Gerar X posições com `python-chess`.
  2. Para cada posição, consultar Stockfish (via UCI) com `depth=12` para obter o gabarito.

* **Decisões**:

  * Usar `python-chess` para criar datasets variados sem depender de arquivos PGN externos.
  * Facilita controle sobre posições e balanceamento.

#### 3.2 Definição do Modelo

* Arquitetura modular de camadas convolucionais (CNN) + camadas fully‑connected.
* Função de perda: cross‑entropy ou KL Divergence para distribuição de movimentos.

#### 3.3 Hiperparâmetros Iniciais

* Learning rate: 1e‑3
* Batch size: 64
* Número de épocas: 10 (ajustável)

---

### 4. Treinamento por Self‑Play (AlphaZero Style)

1. **Configuração de MCTS**

   * Número de simulações por posição: 800 (ajustável).
   * Parâmetro de exploração (Cpuct).

2. **Ciclo de Autojogo**

   * Gerar lotes de jogos completos usando rede+MCTS.
   * Atualizar redes de política e valor com resultados.

3. **Critérios de Parada**

   * Número de iterações autojogo ou convergência de métricas de performance.

##### 4.1. Por que treinar com desafios progressivos?
✅ 1. Porque a rede começa completamente ignorante (pesos aleatórios)
Se você iniciar o treinamento com dados muito complexos:

A rede não tem nenhuma base para entender as jogadas corretas

Ela vai errar feio no começo

Vai gerar sinais de gradiente muito ruidosos, que atrapalham o aprendizado

Treinar com profundidade alta antes de entender o básico é como dar um livro de xadrez avançado para alguém que não sabe mover o cavalo.
✅ 2. Profundidade baixa foca no essencial: “Evite burradas”
Com profundidade 6 ou 8, o Stockfish ainda toma boas decisões, mas ele **prioriza:

Capturas imediatas

Evitar perder peças

Desenvolver rapidamente

Ou seja, os dados dessa fase ajudam a IA a:

Aprender o valor material

Evitar deixar peças penduradas

Começar a construir uma noção de segurança do rei, mobilidade, etc.

Isso é como o fundamento da tática básica humana.

✅ 3. Treinamento progressivo = melhor estabilidade + menos overfitting
Ao treinar em estágios:

Você foca a rede em aprender uma habilidade por vez

A complexidade vai aumentando aos poucos (como um currículo escolar)

Você reduz o risco de overfitting às posições complexas, e aumenta a capacidade de generalização

✅ 4. Profundidade alta pode levar a decisões “contraintuitivas” para a IA iniciante
Exemplo:

O Stockfish, com depth=12, pode sacrificar uma dama por um cheque mate forçado em 8 jogadas.

Para a IA inexperiente, isso parece “errado” (perdeu a peça!) — ela não entende ainda o ganho estratégico de longo prazo.

Isso confunde a rede e atrasa o aprendizado.

Por isso é melhor:
👉 Primeiro ensinar: “dama vale mais que bispo”
👉 Depois: “às vezes, perder dama vale a pena se for por mate”

✅ 5. Evolução progressiva permite reuso e fine-tuning eficiente
Com essa estratégia, você pode:

Treinar uma primeira versão boa rapidamente

Fazer fine-tuning com dados mais profundos sem começar do zero

Reduzir o tempo total de treinamento em até 70%

Evitar explodir seu tempo computacional com profundidade alta no começo



1. **Configuração de MCTS**

   * Número de simulações por posição: 800 (ajustável).
   * Parâmetro de exploração (Cpuct).

2. **Ciclo de Autojogo**

   * Gerar lotes de jogos completos usando rede+MCTS.
   * Atualizar redes de política e valor com resultados.

3. **Critérios de Parada**

   * Número de iterações autojogo ou convergência de métricas de performance.
---

### 5. Avaliação e Benchmark

* Testes periódicos contra Stockfish em diferentes profundidades.
* Métricas: win‑rate, ELO estimado, tempo de resposta.

---

### 6. Infraestrutura

* **Ambiente local**: CPU (inicial).
* **GPU**: planejamento de migração para instância com CUDA.
* **Controle de versões**: Git + GitHub (repositório criado e organizado).

---

### 7. Próximos Passos & Decisões Futuras

* Implementar geração de posições e integração com Stockfish via `python-chess`.
* Testar variações de profundidade e poda (LMR, Null Move).
* Otimizar pipeline de inferência: paralelizar chamadas UCI para Stockfish.
* Configurar logs, checkpoints e monitoramento de métricas.

---

### 8. Changelog (Resumo de Atividades)

* **2025-07-14**:

  * Criamos e organizamos o repositório remoto com Git e GitHub.
  * Decidimos como será feito o treinamento supervisionado e de self-play, definindo fases e justificativas.

* **2025-07-17**:

  * Estruturamos documentação temática em Markdown.
  * Definimos visão geral, arquitetura, treinamento, avaliação, infraestrutura e próximos passos.
