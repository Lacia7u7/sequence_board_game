import React, { useEffect, useState } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { auth, signInAsGuest } from './lib/firebase';

// Simple placeholder pages
const Splash: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const handleGuest = async () => {
    await signInAsGuest();
    navigate('/new');
  };
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-4">{t('welcome')}</h1>
      <button
        className="bg-blue-600 text-white py-2 px-4 rounded mb-2"
        onClick={() => navigate('/login')}
      >
        {t('login')}
      </button>
      <button
        className="bg-green-600 text-white py-2 px-4 rounded mb-2"
        onClick={() => navigate('/signup')}
      >
        {t('signup')}
      </button>
      <button
        className="bg-gray-600 text-white py-2 px-4 rounded"
        onClick={handleGuest}
      >
        {t('guest')}
      </button>
    </div>
  );
};

const Login: React.FC = () => {
  const navigate = useNavigate();
  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO implement login
    navigate('/new');
  };
  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">Login</h2>
      <form onSubmit={handleLogin} className="space-y-2">
        <input className="border p-2 w-full" placeholder="Email" />
        <input className="border p-2 w-full" type="password" placeholder="Password" />
        <button type="submit" className="bg-blue-600 text-white py-2 px-4 rounded">
          Login
        </button>
      </form>
    </div>
  );
};

const Signup: React.FC = () => {
  const navigate = useNavigate();
  const handleSignup = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO implement sign up
    navigate('/new');
  };
  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">Sign Up</h2>
      <form onSubmit={handleSignup} className="space-y-2">
        <input className="border p-2 w-full" placeholder="Username" />
        <input className="border p-2 w-full" placeholder="Display Name" />
        <input className="border p-2 w-full" placeholder="Email" />
        <input className="border p-2 w-full" type="password" placeholder="Password" />
        <button type="submit" className="bg-blue-600 text-white py-2 px-4 rounded">
          Sign Up
        </button>
      </form>
    </div>
  );
};

const NewGame: React.FC = () => {
  const [teams, setTeams] = useState(2);
  const [playersPerTeam, setPlayersPerTeam] = useState(2);
  const [isPublic, setIsPublic] = useState(true);
  const navigate = useNavigate();
  const createGame = async (e: React.FormEvent) => {
    e.preventDefault();
    // call create_match cloud function via Firebase functions or fetch
    // For skeleton we just navigate to lobby with dummy match id
    navigate('/m/dummy');
  };
  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">New Game</h2>
      <form onSubmit={createGame} className="space-y-2">
        <div>
          <label>Teams:</label>
          <select
            value={teams}
            onChange={(e) => setTeams(parseInt(e.target.value))}
            className="border p-2"
          >
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </div>
        <div>
          <label>Players per team:</label>
          <select
            value={playersPerTeam}
            onChange={(e) => setPlayersPerTeam(parseInt(e.target.value))}
            className="border p-2"
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
            <option value={4}>4</option>
          </select>
        </div>
        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={isPublic}
            onChange={(e) => setIsPublic(e.target.checked)}
          />
          <label>Public</label>
        </div>
        <button type="submit" className="bg-green-600 text-white py-2 px-4 rounded">
          Create Game
        </button>
      </form>
    </div>
  );
};

const Lobby: React.FC = () => {
  const navigate = useNavigate();
  // Dummy seats representation
  const seats = [
    { seatIndex: 0, user: { displayName: 'You' }, isReady: true },
    { seatIndex: 1, user: { displayName: 'Player 2' }, isReady: true },
  ];
  const startGame = () => {
    navigate('/m/dummy/play');
  };
  return (
    <div className="p-4">
      <h2 className="text-xl mb-2">Lobby</h2>
      <div className="space-y-2">
        {seats.map((seat) => (
          <div key={seat.seatIndex} className="border p-2 flex justify-between">
            <span>Seat {seat.seatIndex + 1}</span>
            <span>{seat.user?.displayName || 'Empty'}</span>
          </div>
        ))}
      </div>
      <button onClick={startGame} className="mt-4 bg-blue-600 text-white py-2 px-4 rounded">
        Start Game
      </button>
    </div>
  );
};

// A very simple board display
const Game: React.FC = () => {
  const [board, setBoard] = useState<Array<Array<{ card: string; chip: any }>>>([]);
  const [hand, setHand] = useState<string[]>(['AH', 'KD']);
  useEffect(() => {
    // load board from JSON for display
    import('../public/assets/boards/standard_10x10.json').then((json) => {
      const cells = (json as any).cells as string[][];
      const rows = cells.map((row) => row.map((card) => ({ card, chip: null })));
      setBoard(rows);
    });
  }, []);
  const playCard = (r: number, c: number, card: string) => {
    // stub: mark chip and remove card
    setBoard((prev) => {
      const next = prev.map((row) => row.map((cell) => ({ ...cell })));
      next[r][c].chip = { teamIndex: 0 };
      return next;
    });
    setHand((prev) => prev.filter((c2) => c2 !== card));
  };
  return (
    <div className="p-2 flex flex-col items-center">
      <h2 className="text-xl mb-2">Game Board</h2>
      <div className="overflow-auto max-w-full">
        <div
          className="grid"
          style={{ gridTemplateColumns: `repeat(${board.length}, 2rem)` }}
        >
          {board.map((row, r) =>
            row.map((cell, c) => (
              <div
                key={`${r}-${c}`}
                className="w-8 h-8 border flex items-center justify-center text-xs"
                onClick={() => {
                  if (!cell.chip && hand.length > 0) {
                    playCard(r, c, hand[0]);
                  }
                }}
              >
                {cell.chip ? (
                  <span className="text-red-600">●</span>
                ) : (
                  <span>{cell.card}</span>
                )}
              </div>
            )),
          )}
        </div>
      </div>
      <div className="mt-4 flex space-x-2">
        {hand.map((card) => (
          <div key={card} className="border p-1 px-2">
            {card}
          </div>
        ))}
      </div>
    </div>
  );
};

const NotFound: React.FC = () => <div className="p-4">Not Found</div>;

const RoutesComponent: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Splash />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/new" element={<NewGame />} />
      <Route path="/m/:id" element={<Lobby />} />
      <Route path="/m/:id/play" element={<Game />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

export default RoutesComponent;
