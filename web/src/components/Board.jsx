import React, { useState, useCallback } from 'react';
import Chessboard from 'chessboardjsx';
import Chess from 'chess.js';

const Board = () => {
  const [game, setGame] = useState(() => new Chess());
  const [squareStyles, setSquareStyles] = useState({});
  const [pgn, setPgn] = useState('');

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
        body: JSON.stringify({ from: sourceSquare, to: targetSquare }),
      });
      const ai = await res.json();
      const aiMove = g.move(ai);
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

  const undo = () => {
    const g = new Chess(game.fen());
    g.undo();
    g.undo();
    setGame(g);
    setSquareStyles({});
    setPgn(g.pgn());
  };

  return (
    <div>
      <Chessboard position={game.fen()} onDrop={onDrop} squareStyles={squareStyles} />
      <div style={{ marginTop: 10 }}>
        <button onClick={reset}>Restart</button>
        <button onClick={undo}>Undo</button>
      </div>
      <textarea readOnly value={pgn} style={{ width: '100%', height: 100, marginTop: 10 }} />
    </div>
  );
};

export default Board;
