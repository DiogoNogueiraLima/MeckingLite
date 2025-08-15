# MeckingLite

## Objetivo
Projeto para gerar dados de treinamento a partir do Stockfish e treinar uma rede neural de política para sugerir lances de xadrez.

## Requisitos
- Python 3.11+
- [Stockfish](https://stockfishchess.org) instalado
- Dependências do projeto:
  ```bash
  pip install -r requirements.txt
  ```

## Configuração
Edite `utils/config.yaml` e ajuste o caminho do Stockfish e os diretórios de saída:
```yaml
stockfish_path: "/caminho/para/stockfish.exe"
output_data_dir: "data"        # onde salvar o dataset gerado
output_dir: "checkpoints"      # onde salvar os checkpoints do treino
resume: ""                     # opcional: caminho de um .ckpt para retomar
```
Para avaliação, defina também as constantes `STOCKFISH_PATH` e `CHECKPOINT_PATH` no arquivo `scripts/play_WebModel_vs_OurModel.py`.

## Geração de dados
Exemplo de comando para gerar posições com profundidade 6:
```bash
python scripts/generate_data.py --phase v1_depth6 --workers 4 --batch 80
```
Para gerar ambas as fases configuradas use `--phase both`.

## Treinamento
Após gerar os dados, inicie o treinamento supervisionado:
```bash
python scripts/train_supervised.py
```
Os checkpoints serão salvos em `output_dir`. Para retomar, ajuste o campo `resume` no `config.yaml` ou deixe vazio para usar o mais recente automaticamente.

## Avaliação
Configure `STOCKFISH_PATH` e `CHECKPOINT_PATH` em `scripts/play_WebModel_vs_OurModel.py` e execute:
```bash
python scripts/play_WebModel_vs_OurModel.py
```
O script jogará várias partidas e exibirá o placar final.

Simple utilities and neural network experiments for chess.

## API

A small FastAPI server exposes a `/move` endpoint that returns the model's
selected move for a given FEN position.

### Executando

```bash
uvicorn api.server:app
```

Envie uma requisição `POST` com JSON:

```json
{"fen": "FEN AQUI"}
```

A resposta conterá o lance escolhido em notação UCI:

```json
{"move": "e2e4"}
```
