// src/pages/GamePage.tsx
import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { doc, onSnapshot, collection } from 'firebase/firestore';
import { firestore } from '../lib/firebase';
import { useAuth } from '../hooks/useAuth';
import { useShowError } from '../hooks/useShowError';
import Chat from '../components/Chat';
import { callFn } from '../lib/callable';

interface Cell {
  card: string;
  chip: { teamIndex: number } | null;
}

type Axis = 'H' | 'V' | 'D1' | 'D2';
type SeqMeta = { seqId: string; teamIndex: number; axis: Axis; length: number };
type SeqCell = { seqId: string; r: number; c: number };

const isWildJack = (card?: string | null) => card === 'JC' || card === 'JD';
const isRemovalJack = (card?: string | null) => card === 'JH' || card === 'JS';

const getCardImage = (card: string) => {
  if (!card || card === 'BONUS') return null;
  const rank = card.slice(0, -1);
  const suit = card.slice(-1);
  if (!rank || !suit) return null;
  return `/assets/cards/${rank}${suit}.png`;
};

const GamePage: React.FC = () => {
  const { matchId } = useParams<{ matchId: string }>();
  const { user } = useAuth();
  const showError = useShowError();

  const [state, setState] = useState<any>(null);
  const [players, setPlayers] = useState<any>({});
  const [mySeatId, setMySeatId] = useState<string | null>(null);
  const [selectedCard, setSelectedCard] = useState<string | null>(null);

  useEffect(() => {
    if (!matchId) return;

    // State subscription
    const stateRef = doc(firestore, 'matches', matchId, 'state', 'state');
    const unsubState = onSnapshot(stateRef, (snap) => {
      const raw = snap.data();
      if (!raw) {
        setState(null);
        return;
      }
      const board: Cell[][] =
        Array.isArray(raw.board)
          ? raw.board
          : Array.isArray(raw.boardRows)
          ? raw.boardRows.map((r: any) => (Array.isArray(r?.cells) ? r.cells : []))
          : [];

      setState({
        ...raw,
        board,
        // these are arrays in Firestore; still guard to []
        sequencesMeta: Array.isArray(raw.sequencesMeta) ? raw.sequencesMeta : [],
        sequenceCells: Array.isArray(raw.sequenceCells) ? raw.sequenceCells : [],
      });
    });

    // Players subscription (fix: use collection(firestore, ...), not collection(doc(...), ...))
    const playersRef = collection(firestore, 'matches', matchId, 'players');
    const unsubPlayers = onSnapshot(playersRef, (snap) => {
      const data: any = {};
      snap.forEach((d) => (data[d.id] = d.data()));
      setPlayers(data);
      const found = Object.keys(data).find((pid) => data[pid]?.uid === user?.uid);
      setMySeatId(found || null);
    });

    return () => {
      unsubState();
      unsubPlayers();
    };
  }, [matchId, user?.uid]);

  // ---- Safe derived values used by hooks BEFORE any early return ----
  const board: Cell[][] =
    Array.isArray(state?.board) ? state.board : [];
  const rows = board.length;
  const cols = rows > 0 ? board[0].length : 0;

  const sequencesMeta: SeqMeta[] = Array.isArray(state?.sequencesMeta) ? state!.sequencesMeta : [];
  const sequenceCells: SeqCell[] = Array.isArray(state?.sequenceCells) ? state!.sequenceCells : [];

  // Build axes-per-cell overlays from arrays (safe even when empty)
  const axesByCell: Record<string, Axis[]> = useMemo(() => {
    const metaById: Record<string, SeqMeta> = {};
    for (const m of sequencesMeta) {
      if (m && m.seqId) metaById[m.seqId] = m;
    }

    const map: Record<string, Axis[]> = {};
    for (const c of sequenceCells) {
      if (!c || typeof c.r !== 'number' || typeof c.c !== 'number' || !c.seqId) continue;
      const meta = metaById[c.seqId];
      if (!meta) continue;
      const key = `${c.r}-${c.c}`;
      const arr = map[key] ?? [];
      if (!arr.includes(meta.axis)) arr.push(meta.axis);
      map[key] = arr;
    }
    return map;
  }, [sequencesMeta, sequenceCells]);

  // Now it’s safe to early-return UI if state isn’t ready (hooks above have already run every render)
  if (!state) {
    return <div className="p-4">Loading...</div>;
  }

  const hand: string[] = state.hand || (mySeatId ? state.hands?.[mySeatId] || [] : []);
  const currentSeatId: string | null = state.currentSeatId;
  const myTurn = currentSeatId === mySeatId;
  const currentTeamIndex = mySeatId ? players[mySeatId]?.teamIndex : null;

  // ---- Interactions ----
  const playCardOnCell = async (r: number, c: number) => {
    if (!myTurn || !selectedCard || !mySeatId) return;
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;

    const cell = board[r]?.[c];
    if (!cell) return;

    // Wild Jack: JC/JD -> must place on free, non-BONUS
    if (isWildJack(selectedCard)) {
      if (cell.card === 'BONUS') {
        showError({ code: 'ERR_INVALID_JACK_USE' });
        return;
      }
      if (cell.chip != null) {
        showError({ code: 'ERR_TARGET_OCCUPIED' });
        return;
      }
      try {
        await callFn('submit_move', {
          matchId,
          seatId: mySeatId,
          type: 'wild',
          card: selectedCard,
          target: { r, c },
          removed: null,
        });
        setSelectedCard(null);
      } catch (err) {
        showError(err);
      }
      return;
    }

    // Removal Jack: JH/JS -> click opponent chip to remove
    if (isRemovalJack(selectedCard)) {
      const chip = cell.chip;
      if (!chip) {
        showError({ code: 'ERR_NOT_MATCHING_CARD' }); // empty cell
        return;
      }
      if (chip.teamIndex === currentTeamIndex) {
        showError({ code: 'ERR_CANNOT_REMOVE_OWN_CHIP' });
        return;
      }
      try {
        await callFn('submit_move', {
          matchId,
          seatId: mySeatId,
          type: 'jack-remove',
          card: selectedCard,
          target: null,
          removed: { r, c },
        });
        setSelectedCard(null);
      } catch (err) {
        showError(err);
      }
      return;
    }

    // Normal card play
    if (cell.chip != null) {
      showError({ code: 'ERR_TARGET_OCCUPIED' });
      return;
    }
    try {
      await callFn('submit_move', {
        matchId,
        seatId: mySeatId,
        type: 'play',
        card: selectedCard,
        target: { r, c },
        removed: null,
      });
      setSelectedCard(null);
    } catch (err) {
      showError(err);
    }
  };

  const burnSelected = async () => {
    if (!selectedCard || !mySeatId) return;
    if (isWildJack(selectedCard) || isRemovalJack(selectedCard)) {
      // Jacks cannot be burned
      return;
    }
    try {
      await callFn('submit_move', {
        matchId,
        seatId: mySeatId,
        type: 'burn',
        card: selectedCard,
      });
      setSelectedCard(null);
    } catch (err) {
      showError(err);
    }
  };

  return (
    <div className="p-4 max-w-5xl mx-auto grid gap-4 grid-cols-1 md:grid-cols-4">
      <div className="md:col-span-3 flex flex-col items-center">
        <h2 className="text-xl font-bold mb-2">Game Board</h2>

        {cols > 0 ? (
          <div className="overflow-auto border rounded shadow" style={{ maxHeight: '70vh' }}>
            <div className="grid relative" style={{ gridTemplateColumns: `repeat(${cols}, 2.5rem)` }}>
              {board.map((row, r) =>
                row.map((cell, c) => {
                  const cardImg = getCardImage(cell?.card);
                  const isBonus = cell?.card === 'BONUS';
                  const isMine = cell?.chip?.teamIndex === currentTeamIndex;
                  const seqAxes = axesByCell[`${r}-${c}`] ?? [];

                  return (
                    <div
                      key={`${r}-${c}`}
                      className={`relative w-10 h-12 border flex items-center justify-center ${
                        isBonus ? 'bg-yellow-200/60' : 'bg-white'
                      } ${cell?.chip ? 'cursor-default' : myTurn ? 'cursor-pointer hover:bg-indigo-50' : 'cursor-default'}`}
                      onClick={() => myTurn && playCardOnCell(r, c)}
                    >
                      {/* card image / text */}
                      {cardImg ? (
                        <img src={cardImg} alt={cell.card} className="w-full h-full object-contain pointer-events-none" />
                      ) : (
                        <span className="text-xs select-none">{isBonus ? '★' : cell?.card}</span>
                      )}

                      {/* chip */}
                      {cell?.chip && (
                        <span
                          className={`absolute w-4 h-4 rounded-full ring-2 ring-white ${
                            isMine ? 'bg-blue-500' : cell.chip.teamIndex === 1 ? 'bg-green-500' : 'bg-red-500'
                          }`}
                          style={{ top: '2px', right: '2px' }}
                          title={`Team ${cell.chip.teamIndex + 1}`}
                        />
                      )}

                      {/* sequence overlays (thin bars) */}
                      {seqAxes.map((ax) => {
                        const overlayKey = `seq-${r}-${c}-${ax}`;
                        if (ax === 'H') {
                          return (
                            <span
                              key={overlayKey}
                              className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-0.5 bg-indigo-500/70 pointer-events-none"
                            />
                          );
                        } else if (ax === 'V') {
                          return (
                            <span
                              key={overlayKey}
                              className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-0.5 bg-indigo-500/70 pointer-events-none"
                            />
                          );
                        } else if (ax === 'D1') {
                          return (
                            <span
                              key={overlayKey}
                              className="absolute w-[140%] h-0.5 bg-indigo-500/70 rotate-45 pointer-events-none"
                              style={{ left: '-20%', top: '50%' }}
                            />
                          );
                        } else {
                          return (
                            <span
                              key={overlayKey}
                              className="absolute w-[140%] h-0.5 bg-indigo-500/70 -rotate-45 pointer-events-none"
                              style={{ left: '-20%', top: '50%' }}
                            />
                          );
                        }
                      })}
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
        <div className="mt-4 flex flex-wrap gap-2 justify-center max-w-full">
          {hand.map((card, i) => {
            const img = getCardImage(card);
            const selected = card === selectedCard;
            return (
              <button
                key={`${card}-${i}`}
                className={`border rounded overflow-hidden ${selected ? 'ring-4 ring-indigo-500' : ''}`}
                onClick={() => setSelectedCard(card)}
                title={isWildJack(card) ? 'Wild Jack' : isRemovalJack(card) ? 'Cut Jack' : card}
              >
                {img ? (
                  <img src={img} alt={card} className="w-12 h-16 object-contain" />
                ) : (
                  <div className="w-12 h-16 flex items-center justify-center">{card}</div>
                )}
              </button>
            );
          })}
        </div>

        {/* Jack badge + Burn */}
        {selectedCard && (
          <div className="mt-2 flex items-center gap-3">
            {isWildJack(selectedCard) && <span className="text-xs px-2 py-1 rounded bg-indigo-100 text-indigo-700">Wild Jack</span>}
            {isRemovalJack(selectedCard) && <span className="text-xs px-2 py-1 rounded bg-rose-100 text-rose-700">Cut Jack</span>}

            <button
              onClick={burnSelected}
              disabled={isWildJack(selectedCard) || isRemovalJack(selectedCard)}
              className={`py-1 px-2 rounded text-sm ${
                isWildJack(selectedCard) || isRemovalJack(selectedCard)
                  ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
                  : 'bg-gray-600 text-white hover:bg-gray-700'
              }`}
              title={isWildJack(selectedCard) || isRemovalJack(selectedCard) ? 'Jacks cannot be burned' : 'Burn card'}
            >
              Burn Card
            </button>
          </div>
        )}
      </div>

      {/* Right column: players + chat */}
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
