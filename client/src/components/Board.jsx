import React, { useState, useCallback } from 'react';
import Chessboard from 'chessboardjsx';
import { Chess } from 'chess.js';

const Board = () => {
  const [game, setGame] = useState(() => new Chess());
  const [squareStyles, setSquareStyles] = useState({});
  const [pgn, setPgn] = useState('');
  const [orientation, setOrientation] = useState('white');
  const [gameStatus, setGameStatus] = useState(null); // Para controlar final de jogo
  const [isAiThinking, setIsAiThinking] = useState(false); // Para indicar quando IA está pensando

  const highlight = (move) => ({
    [move.from]: { background: 'rgba(255,255,0,0.4)' },
    [move.to]: { background: 'rgba(255,255,0,0.4)' },
  });

  const onDrop = useCallback(async ({ sourceSquare, targetSquare }) => {
    // Não permitir jogadas se o jogo acabou ou se a IA está pensando
    if (gameStatus || isAiThinking) return;

    const g = new Chess(game.fen());
    const move = g.move({ from: sourceSquare, to: targetSquare, promotion: 'q' });
    if (!move) return;

    setGame(g);
    setSquareStyles(highlight(move));
    setPgn(g.pgn());

    // Verificar se o jogador ganhou (IA não pode jogar)
    if (g.isGameOver()) {
      if (g.isCheckmate()) {
        setGameStatus({ type: 'win', message: 'Você venceu! Xeque-mate!' });
      } else if (g.isStalemate()) {
        setGameStatus({ type: 'draw', message: 'Empate por afogamento!' });
      } else if (g.isInsufficientMaterial()) {
        setGameStatus({ type: 'draw', message: 'Empate por material insuficiente!' });
      } else {
        setGameStatus({ type: 'draw', message: 'Empate!' });
      }
      return;
    }

    try {
      setIsAiThinking(true); // IA começou a pensar
      const res = await fetch('/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: g.fen() }),
      });
      const response = await res.json();

      // Verificar se a IA detectou final de jogo
      if (response.game_over) {
        if (response.result === 'checkmate') {
          if (response.winner === orientation) {
            setGameStatus({ type: 'win', message: 'Você venceu! A IA não tem jogadas!' });
          } else {
            setGameStatus({ type: 'lose', message: 'Você perdeu! Xeque-mate!' });
          }
        } else {
          setGameStatus({ type: 'draw', message: `Empate por ${response.result}!` });
        }
        return;
      }

      // Jogada normal da IA
      if (response.move) {
        const aiMove = g.move(response.move);
        if (aiMove) {
          setGame(new Chess(g.fen()));
          setSquareStyles(highlight(aiMove));
          setPgn(g.pgn());

          // Verificar se a IA ganhou após sua jogada
          const updatedGame = new Chess(g.fen());
          if (updatedGame.isGameOver()) {
            if (updatedGame.isCheckmate()) {
              setGameStatus({ type: 'lose', message: 'Você perdeu! Xeque-mate!' });
            } else if (updatedGame.isStalemate()) {
              setGameStatus({ type: 'draw', message: 'Empate por afogamento!' });
            } else {
              setGameStatus({ type: 'draw', message: 'Empate!' });
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsAiThinking(false); // IA terminou de pensar
    }
  }, [game, gameStatus, orientation, isAiThinking]);

  const reset = () => {
    const g = new Chess();
    setGame(g);
    setSquareStyles({});
    setPgn('');
    setGameStatus(null);
    setIsAiThinking(false);
  };

  const trade = async () => {
    // Reinicia o jogo
    const g = new Chess();
    setGame(g);
    setSquareStyles({});
    setPgn('');
    setGameStatus(null);
    
    // Inverte a orientação do tabuleiro
    const newOrientation = orientation === 'white' ? 'black' : 'white';
    setOrientation(newOrientation);
    
    // Se o jogador escolheu jogar de pretas, a IA (brancas) deve fazer a primeira jogada
    if (newOrientation === 'black') {
      try {
        setIsAiThinking(true);
        const res = await fetch('/move', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fen: g.fen() }), // Posição inicial
        });
        const response = await res.json();
        
        if (response.move) {
          const aiMove = g.move(response.move);
          if (aiMove) {
            setGame(new Chess(g.fen()));
            setSquareStyles(highlight(aiMove));
            setPgn(g.pgn());
          }
        }
      } catch (err) {
        console.error('Erro ao solicitar jogada da IA:', err);
      } finally {
        setIsAiThinking(false);
      }
    }
  };

  return (
    <div className="board-container">
      <div className="board-card">
        <Chessboard
          width={480}
          position={game.fen()}
          onDrop={onDrop}
          squareStyles={squareStyles}
          orientation={orientation}
        />
        {gameStatus && (
          <div className={`game-status ${gameStatus.type}`}>
            <h2>{gameStatus.message}</h2>
            <button onClick={reset} className="restart-btn">
              Jogar Novamente
            </button>
          </div>
        )}
        {isAiThinking && (
          <div className="ai-thinking">
            <div className="thinking-spinner"></div>
            <p>IA está pensando...</p>
          </div>
        )}
      </div>
      <div className="controls">
        <button onClick={reset}>Restart</button>
        <button onClick={trade}>Trade</button>
      </div>
      <textarea className="pgn" readOnly value={pgn} />
    </div>
  );
};

export default Board;
