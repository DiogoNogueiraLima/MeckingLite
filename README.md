# MeckingLite

## 🎯 Objetivo
Projeto de xadrez com IA que permite jogar contra um modelo de rede neural treinado. Combina desenvolvimento web moderno (React + NestJS) com inteligência artificial em Python para criar uma experiência interativa de xadrez.

## 🏗️ Arquitetura

```
├── client/          # Frontend React com Vite
├── server/          # Backend NestJS (gateway)
├── model/           # Rede neural e serviços Python
├── scripts/         # Scripts de treinamento e avaliação
└── utils/           # Configurações
```

### Fluxo de dados:
```
Client (React) → Server (NestJS) → Model (Python) → Resposta
```

## 🛠️ Tecnologias

### Frontend
- **React 18** - Interface de usuário
- **Vite** - Build tool e dev server
- **chessboardjsx** - Componente do tabuleiro
- **chess.js** - Lógica do jogo

### Backend
- **NestJS** - Framework Node.js enterprise
- **TypeScript** - Tipagem estática
- **Express** - Servidor HTTP

### IA/Modelo
- **PyTorch** - Framework de deep learning
- **python-chess** - Manipulação de posições
- **NumPy** - Computação numérica
- **PyYAML** - Configurações

## 📦 Instalação

### Pré-requisitos
- **Node.js 18+**
- **Python 3.9+**
- **npm ou yarn**

### 1. Dependências Python (globais)
```bash
pip3 install torch chess python-chess pyyaml numpy
```

### 2. Servidor NestJS
```bash
cd server
npm install
```

### 3. Cliente React
```bash
cd client
npm install
```

## 🚀 Como rodar

### 1. Servidor (porta 3000)
```bash
cd server
npm run start:dev
```

### 2. Frontend (porta 5173)
```bash
cd client
npm run dev
```

### 3. Acessar aplicação
Abra: **http://localhost:5173**

## 🎮 Como usar

1. **Faça um lance** - Arraste uma peça no tabuleiro
2. **Aguarde a IA** - O modelo calculará automaticamente o próximo lance
3. **Continue jogando** - A IA responderá a cada movimento
4. **Botões disponíveis:**
   - **Restart** - Reinicia a partida
   - **Trade** - Inverte cores e reinicia

## 🧠 Como funciona o modelo

### Modelo de IA
A cada lance do usuário:
1. **Frontend** envia posição atual (FEN) para o servidor
2. **Servidor NestJS** chama script Python
3. **Script Python** carrega modelo treinado do disco
4. **Modelo** analisa posição e calcula melhor lance
5. **Resposta** retorna ao frontend que atualiza o tabuleiro

### Vantagens desta abordagem:
- ✅ **Simples** - Cada requisição é independente
- ✅ **Econômico** - Só usa RAM quando necessário  
- ✅ **Flexível** - Fácil trocar modelos
- ✅ **Familiar** - Controle total em TypeScript

## 🧪 Treinamento do modelo (opcional)

### 1. Configuração
Edite `utils/config.yaml`:
```yaml
stockfish_path: "/caminho/para/stockfish"
output_data_dir: "data"
output_dir: "checkpoints"
```

### 2. Gerar dados
```bash
python scripts/generate_data.py --phase v1_depth6 --workers 4
```

### 3. Treinar modelo
```bash
python scripts/train_supervised.py
```

### 4. Avaliar modelo
```bash
python scripts/play_WebModel_vs_OurModel.py
```

## 📁 Estrutura de arquivos

```
MeckingLite/
├── client/
│   ├── src/
│   │   ├── components/Board.jsx    # Componente do tabuleiro
│   │   ├── App.jsx                 # App principal
│   │   └── main.jsx               # Entry point
│   ├── package.json
│   └── vite.config.js             # Proxy para servidor
├── server/
│   ├── src/
│   │   ├── move.controller.ts     # Endpoint POST /move
│   │   ├── app.module.ts          # Configuração NestJS
│   │   └── main.ts               # Servidor na porta 3000
│   └── package.json
├── model/
│   ├── inference_service.py       # Serviço de predição
│   ├── network.py                 # Arquitetura da rede
│   ├── dataset.py                 # Processamento de dados
│   └── heuristics/               # Funções de avaliação
├── scripts/                      # Scripts de ML
└── utils/config.yaml            # Configurações globais
```

## 🐛 Troubleshooting

### Porta 3000 ocupada
```bash
lsof -n -i :3000
kill <PID>
```

### Erro de módulo Python
Certifique-se de ter as dependências instaladas:
```bash
pip3 install torch chess python-chess pyyaml numpy
```

### Frontend em branco
1. Verifique se o servidor está rodando na porta 3000
2. Abra DevTools → Console para ver erros
3. Verifique se `client/node_modules` existe

## 🎯 Próximos passos

- [ ] Melhorar UI/UX do tabuleiro
- [ ] Adicionar histórico de partidas  
- [ ] Implementar diferentes níveis de dificuldade
- [ ] Otimizar performance do modelo
- [ ] Adicionar análise de posições

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request
