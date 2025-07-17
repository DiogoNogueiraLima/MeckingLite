📅 PLANEJAMENTO SEMANAL
✅ Fase 1 — Conclusão da Geração de Dados (hoje e amanhã)
🔹 Objetivo: Gerar um dataset de alta qualidade com as top-3 jogadas do Stockfish e seus scores.

Tarefas:

 Adaptar stockfish_data_gen.py para salvar as top-3 jogadas com score (✅ já feito!)

 Validar os dados com script de visualização simples

 Criar função que transforma scores em distribuição de probabilidade (target)

✅ Fase 2 — Treinamento Supervisionado Inicial (próximos 2 dias)
🔹 Objetivo: Treinar uma rede simples com as top-3 jogadas como target probabilístico

Tarefas:

 Criar train_supervised.py com rede básica (PyTorch)

 Loss function baseada em cross-entropy com distribuição de scores

 Métricas: Top-1, Top-3 accuracy, KL divergence

 Visualização de curva de perda e acurácia com Matplotlib

✅ Fase 3 — Interface Gráfica (fim de semana)
🔹 Objetivo: Ver a IA jogando, analisar jogadas e progresso em tempo real

Tarefas:

 Construir interface com tkinter ou PyQt e python-chess

 Mostrar jogadas da IA vs. Stockfish em tempo real

 Carregar modelo salvo e permitir partida contra o humano (opcional)

✅ Fase 4 — Avaliação e Validação do Modelo (início da próxima semana)
🔹 Objetivo: Validar se a rede está aprendendo padrões de jogo

Tarefas:

 Aplicar a rede em posições do dataset que não foram vistas

 Comparar jogadas sugeridas com as top do Stockfish

 Exportar gráficos de desempenho (centipawn loss médio por jogada)

✅ Fase 5 — Self-Play e Reinforcement Learning (semana que vem)
🔹 Objetivo: A IA treina contra si mesma e se autoaperfeiçoa

Tarefas:

 Criar módulo de self-play com MCTS básico

 Usar rede como política + avaliação de posição

 Atualizar rede a cada X partidas com aprendizado por reforço