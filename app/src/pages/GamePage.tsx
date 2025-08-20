import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { doc, onSnapshot, collection } from 'firebase/firestore';
import { httpsCallable } from 'firebase/functions';
import { firestore, functions } from '../lib/firebase';
import { useAuth } from '../hooks/useAuth';
import Chat from '../components/Chat';

interface Cell {
  card: string;
  chip: { teamIndex: number } | null;
}

const getCardImage = (card: string) => {
  // files live under /assets/cards/{RANK}{SUIT}.JPG  e.g., AS.JPG
  if (!card || card === 'BONUS') return null;
  const rank = card.slice(0, -1);
  const suit = card.slice(-1);
  if (!rank || !suit) return null;
  return `/assets/cards/${rank}${suit}.png`;
};

const GamePage: React.FC = () => {
  const { matchId } = useParams<{ matchId: string }>();
  const { user } = useAuth();
  const [state, setState] = useState<any>(null);
  const [players, setPlayers] = useState<any>({});
  const [mySeatId, setMySeatId] = useState<string | null>(null);
  const [selectedCard, setSelectedCard] = useState<string | null>(null);
  const [pendingJackAction, setPendingJackAction] = useState<{ card: string; type: 'wild' | 'remove' } | null>(null);

  useEffect(() => {
    if (!matchId) return;

    const stateRef = doc(firestore, 'matches', matchId, 'state', 'state');
    const unsubState = onSnapshot(stateRef, (snap) => {
      const raw = snap.data();
      if (!raw) {
        setState(null);
        return;
      }
      // 🔧 Derivar board 2D desde boardRows para que el resto del componente no cambie
      const board: Cell[][] =
        Array.isArray(raw.board) ? raw.board :
        Array.isArray(raw.boardRows) ? raw.boardRows.map((r: any) => Array.isArray(r?.cells) ? r.cells : []) :
        [];

      setState({ ...raw, board }); // guardamos 'board' listo para usar
    });

    const playersRef = collection(doc(firestore, 'matches', matchId), 'players');
    const unsubPlayers = onSnapshot(playersRef, (snap) => {
      const data: any = {};
      snap.forEach((d) => (data[d.id] = d.data()));
      setPlayers(data);
      const found = Object.keys(data).find((pid) => data[pid].uid === user?.uid);
      setMySeatId(found || null);
    });

    return () => {
      unsubState();
      unsubPlayers();
    };
  }, [matchId, user?.uid]);

  if (!state) return <div className="p-4">Loading...</div>;

  const board: Cell[][] = Array.isArray(state.board) ? state.board : [];
  const cols = board[0]?.length ?? 0;

  const hand: string[] = state.hand || (mySeatId ? state.hands?.[mySeatId] || [] : []);
  const currentSeatId: string | null = state.currentSeatId;
  const myTurn = currentSeatId === mySeatId;
  const currentTeamIndex = mySeatId ? players[mySeatId]?.teamIndex : null;

  const playCardOnCell = async (r: number, c: number) => {
    if (!selectedCard || !mySeatId) return;
    let moveType: string = 'play';
    let removed: any = null;
    let target: any = { r, c };

    if (selectedCard.startsWith('J')) {
      if (!pendingJackAction) {
        setPendingJackAction({ card: selectedCard, type: 'wild' });
        return;
      } else {
        moveType = pendingJackAction.type === 'wild' ? 'wild' : 'jack-remove';
        if (pendingJackAction.type === 'remove') {
          removed = { r, c };
          target = null;
        }
      }
    }

    try {
      const submit = httpsCallable(functions, 'submit_move');
      await submit({ matchId, seatId: mySeatId, type: moveType, card: selectedCard, target, removed });
      setSelectedCard(null);
      setPendingJackAction(null);
    } catch (err: any) {
      console.error(err);
      alert(err?.message || 'Move failed');
    }
  };

  const burnSelected = async () => {
    if (!selectedCard || !mySeatId) return;
    try {
      const submit = httpsCallable(functions, 'submit_move');
      await submit({ matchId, seatId: mySeatId, type: 'burn', card: selectedCard });
      setSelectedCard(null);
    } catch (err: any) {
      console.error(err);
      alert(err?.message || 'Burn failed');
    }
  };

  return (
    <div className="p-4 max-w-5xl mx-auto grid gap-4 grid-cols-1 md:grid-cols-4">
      <div className="md:col-span-3 flex flex-col items-center">
        <h2 className="text-xl font-bold mb-2">Game Board</h2>

        {cols > 0 ? (
          <div className="overflow-auto border rounded shadow" style={{ maxHeight: '70vh' }}>
            <div className="grid" style={{ gridTemplateColumns: `repeat(${cols}, 2.5rem)` }}>
              {board.map((row, r) =>
                row.map((cell, c) => {
                  const cardImg = getCardImage(cell?.card);
                  const isBonus = cell?.card === 'BONUS';
                  return (
                    <div
                      key={`${r}-${c}`}
                      className={`w-10 h-12 border flex items-center justify-center relative ${isBonus ? 'bg-yellow-200' : 'bg-white'}`}
                      onClick={() => myTurn && playCardOnCell(r, c)}
                    >
                      {cardImg ? (
                        <img src={cardImg} alt={cell.card} className="w-full h-full object-contain" />
                      ) : (
                        <span className="text-xs">{isBonus ? '★' : cell?.card}</span>
                      )}
                      {cell?.chip && (
                        <span
                          className={`absolute w-4 h-4 rounded-full ${
                            cell.chip.teamIndex === 0
                              ? 'bg-blue-500'
                              : cell.chip.teamIndex === 1
                              ? 'bg-green-500'
                              : 'bg-red-500'
                          }`}
                          style={{ top: '2px', right: '2px' }}
                        />
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-600">Board not ready yet…</div>
        )}

        {/* Hand */}
        <div className="mt-4 flex space-x-2 flex-wrap justify-center max-w-full">
          {hand.map((card) => {
            const img = getCardImage(card);
            const selected = card === selectedCard;
            return (
              <div
                key={card + Math.random()}
                className={`border rounded overflow-hidden cursor-pointer ${selected ? 'ring-4 ring-indigo-500' : ''}`}
                onClick={() => setSelectedCard(card)}
              >
                {img ? (
                  <img src={img} alt={card} className="w-12 h-16 object-contain" />
                ) : (
                  <div className="w-12 h-16 flex items-center justify-center">{card}</div>
                )}
              </div>
            );
          })}
        </div>

        {selectedCard && (
          <div className="mt-2 flex items-center space-x-2">
            {selectedCard.startsWith('J') && !pendingJackAction && (
              <>
                <button
                  onClick={() => setPendingJackAction({ card: selectedCard, type: 'wild' })}
                  className="py-1 px-2 bg-blue-600 text-white rounded text-sm"
                >
                  Use as Wild
                </button>
                <button
                  onClick={() => setPendingJackAction({ card: selectedCard, type: 'remove' })}
                  className="py-1 px-2 bg-red-600 text-white rounded text-sm"
                >
                  Remove Chip
                </button>
              </>
            )}
            <button onClick={burnSelected} className="py-1 px-2 bg-gray-500 text-white rounded text-sm">
              Burn Card
            </button>
          </div>
        )}
      </div>

      <div className="md:col-span-1 flex flex-col space-y-4">
        <div className="border rounded shadow p-2">
          <h3 className="font-semibold mb-1">Players</h3>
          <ul className="space-y-1 text-sm">
            {Object.entries(players)
              .sort((a: any, b: any) => a[1].seatIndex - b[1].seatIndex)
              .map(([pid, p]: any) => (
                <li key={pid} className={`${pid === currentSeatId ? 'bg-yellow-100' : ''} p-1 flex justify-between`}>
                  <span>
                    {p.displayName || 'AI'} {pid === mySeatId && '(You)'}
                  </span>
                  <span className="text-xs text-gray-500">Team {p.teamIndex + 1}</span>
                </li>
              ))}
          </ul>
          {state.winners && state.winners.length > 0 && (
            <div className="mt-2 text-green-700 font-bold">Winner: Team {state.winners[0] + 1}</div>
          )}
        </div>
        <Chat matchId={matchId!} teamIndex={currentTeamIndex} />
      </div>
    </div>
  );
};

export default GamePage;
