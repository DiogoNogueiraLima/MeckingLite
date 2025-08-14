import React, { useState, useCallback } from 'react';
import Chessboard from 'chessboardjsx';
import { Chess } from 'chess.js';

const Board = () => {
  const [game, setGame] = useState(() => new Chess());
  const [squareStyles, setSquareStyles] = useState({});
  const [pgn, setPgn] = useState('');
  const [orientation, setOrientation] = useState('white');

  const highlight = (move) => ({
    [move.from]: { background: 'rgba(255,255,0,0.4)' },
    [move.to]: { background: 'rgba(255,255,0,0.4)' },
  });

  const onDrop = useCallback(async ({ sourceSquare, targetSquare }) => {
    const g = new Chess(game.fen());
    const move = g.move({ from: sourceSquare, to: targetSquare, promotion: 'q' });
    if (!move) return;

    setGame(g);
    setSquareStyles(highlight(move));
    setPgn(g.pgn());

    try {
      const res = await fetch('/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen: g.fen() }),
      });
      const { move } = await res.json();
      const aiMove = g.move(move);
      if (aiMove) {
        setGame(new Chess(g.fen()));
        setSquareStyles(highlight(aiMove));
        setPgn(g.pgn());
      }
    } catch (err) {
      console.error(err);
    }
  }, [game]);

  const reset = () => {
    const g = new Chess();
    setGame(g);
    setSquareStyles({});
    setPgn('');
  };

  const trade = () => {
    // Reinicia o jogo
    const g = new Chess();
    setGame(g);
    setSquareStyles({});
    setPgn('');
    // Inverte a orientação do tabuleiro
    setOrientation(orientation === 'white' ? 'black' : 'white');
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
