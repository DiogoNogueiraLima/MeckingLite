# MeckingLite

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
