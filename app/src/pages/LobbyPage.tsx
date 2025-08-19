import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { httpsCallable } from 'firebase/functions';
import { doc, collection, onSnapshot, getDoc } from 'firebase/firestore';
import { functions, firestore } from '../lib/firebase';
import { useAuth } from '../hooks/useAuth';
import Chat from '../components/Chat';

interface PlayerSeat {
  seatIndex: number;
  teamIndex: number;
  uid: string | null;
  displayName: string | null;
  isAgent: boolean;
  seatCode: string;
}

const LobbyPage: React.FC = () => {
  const { matchId } = useParams<{ matchId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [match, setMatch] = useState<any>(null);
  const [players, setPlayers] = useState<Record<string, PlayerSeat>>({});
  const [seatCodeInput, setSeatCodeInput] = useState('');
  const [joinCodeInput, setJoinCodeInput] = useState('');
  const [displayName, setDisplayName] = useState('');
  // track if user has seat
  const [mySeatId, setMySeatId] = useState<string | null>(null);
  useEffect(() => {
    if (!matchId) return;
    const matchRef = doc(firestore, 'matches', matchId);
    const unsubMatch = onSnapshot(matchRef, (snap) => {
      setMatch(snap.data());
    });
    const playersRef = collection(matchRef, 'players');
    const unsubPlayers = onSnapshot(playersRef, (snap) => {
      const data: Record<string, PlayerSeat> = {};
      snap.forEach((docSnap) => {
        data[docSnap.id] = docSnap.data() as unknown as PlayerSeat;
      });
      setPlayers(data);
      // Find my seat id by uid
      const found = Object.keys(data).find((pid) => data[pid].uid === user?.uid);
      setMySeatId(found || null);
    });
    return () => {
      unsubMatch();
      unsubPlayers();
    };
  }, [matchId, user?.uid]);
  const handleJoin = async () => {
    if (!matchId) return;
    try {
      const joinMatch = httpsCallable(functions, 'join_match');
      const res: any = await joinMatch({
        matchId,
        joinCode: match?.security?.joinCode || null,
        seatCode: seatCodeInput,
        displayName: displayName || user?.displayName,
      });
      console.log(res.data);
    } catch (err) {
      console.error(err);
      alert('Error joining match');
    }
  };
  const handleStart = async () => {
    if (!matchId) return;
    try {
      const start = httpsCallable(functions, 'start_if_ready');
      const res: any = await start({ matchId });
      console.log(res.data);
    } catch (err) {
      console.error(err);
      alert('Error starting match');
    }
  };
  if (!match) return <div className="p-4">Loading...</div>;
  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Lobby</h2>
      <p>Status: {match.status}</p>
      {!mySeatId && (
        <div className="bg-white p-4 rounded shadow mb-4">
          <h3 className="text-lg font-semibold mb-2">Join this game</h3>
          {!match.security?.public && (
            <div className="mb-2">
              <label className="block">Join Code:</label>
              <input
                type="text"
                value={joinCodeInput}
                onChange={(e) => setJoinCodeInput(e.target.value)}
                className="border p-2 rounded w-full"
              />
            </div>
          )}
          <div className="mb-2">
            <label className="block">Seat Code:</label>
            <input
              type="text"
              value={seatCodeInput}
              onChange={(e) => setSeatCodeInput(e.target.value)}
              className="border p-2 rounded w-full"
            />
          </div>
          <div className="mb-2">
            <label className="block">Display Name (optional):</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="border p-2 rounded w-full"
            />
          </div>
          <button
            onClick={handleJoin}
            className="py-2 px-4 bg-indigo-600 text-white rounded hover:bg-indigo-700"
          >
            Join
          </button>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h3 className="text-lg font-semibold mb-2">Players</h3>
          <ul className="space-y-2">
            {Object.entries(players)
              .sort((a, b) => a[1].seatIndex - b[1].seatIndex)
              .map(([pid, seat]) => (
                <li
                  key={pid}
                  className={`p-2 border rounded flex justify-between ${pid === mySeatId ? 'bg-green-100' : ''}`}
                >
                  <span>
                    Seat {seat.seatIndex + 1} (Team {seat.teamIndex + 1})
                  </span>
                    <span>
                    {seat.displayName || 'Available'} {seat.isAgent && '(AI)'}
                    </span>
                </li>
              ))}
          </ul>
        </div>
        <div>
          <Chat matchId={matchId!} teamIndex={players[mySeatId || '']?.teamIndex ?? null} />
        </div>
      </div>
      {match.status === 'active' && (
        <button
          onClick={() => navigate(`/m/${matchId}/play`)}
          className="mt-4 py-2 px-4 bg-green-600 text-white rounded"
        >
          Go to Game
        </button>
      )}
      {match.status === 'lobby' && (
        <button
          onClick={handleStart}
          className="mt-4 py-2 px-4 bg-blue-600 text-white rounded"
        >
          Start Game
        </button>
      )}
    </div>
  );
};

export default LobbyPage;
