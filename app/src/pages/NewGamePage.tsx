import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { httpsCallable } from 'firebase/functions';
import { functions, storage } from '../lib/firebase';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';

const NewGamePage: React.FC = () => {
  const navigate = useNavigate();
  const [teams, setTeams] = useState(2);
  const [playersPerTeam, setPlayersPerTeam] = useState(2);
  const [allowAdvancedJack, setAllowAdvancedJack] = useState(false);
  const [allowDraws, setAllowDraws] = useState(false);
  const [turnSeconds, setTurnSeconds] = useState(60);
  const [totalMinutes, setTotalMinutes] = useState(30);
  const [isPublic, setIsPublic] = useState(true);
  const [weightsFile, setWeightsFile] = useState<File | null>(null);
  const [creating, setCreating] = useState(false);
  const [createdInfo, setCreatedInfo] = useState<any>(null);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    try {
      let weightsURL: string | null = null;
      if (weightsFile) {
        const storageRef = ref(storage, `weights/tmp/${Date.now()}_${weightsFile.name}`);
        await uploadBytes(storageRef, weightsFile);
        weightsURL = await getDownloadURL(storageRef);
      }
      const createMatch = httpsCallable(functions, 'create_match');
      const result: any = await createMatch({
        config: {
          teams,
          playersPerTeam,
          allowAdvancedJack,
          allowDraws,
          turnSeconds,
          totalMinutes,
        },
        public: isPublic,
        agentWeightsURL: weightsURL,
      });
      setCreatedInfo(result.data);
    } catch (err) {
      console.error(err);
      alert('Error creating match');
    } finally {
      setCreating(false);
    }
  };
  if (createdInfo) {
    return (
      <div className="p-4">
        <h2 className="text-2xl font-bold mb-4">Match Created</h2>
        <p>Match ID: {createdInfo.matchId}</p>
        <p>Join Code: {createdInfo.joinCode}</p>
        <p>Seat Codes:</p>
        <ul className="list-disc list-inside">
          {Object.entries(createdInfo.seatCodes).map(([id, code]) => (
            <li key={id}>
              Seat {parseInt(id) + 1}: {code}
            </li>
          ))}
        </ul>
        <button
          className="mt-4 py-2 px-4 bg-indigo-600 text-white rounded"
          onClick={() => navigate(`/m/${createdInfo.matchId}`)}
        >
          Go to Lobby
        </button>
      </div>
    );
  }
  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">Create New Game</h2>
      <form onSubmit={handleSubmit} className="space-y-3 max-w-md">
        <div>
          <label className="block font-medium">Teams:</label>
          <select
            value={teams}
            onChange={(e) => setTeams(parseInt(e.target.value))}
            className="border p-2 rounded w-full"
          >
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </div>
        <div>
          <label className="block font-medium">Players per team:</label>
          <select
            value={playersPerTeam}
            onChange={(e) => setPlayersPerTeam(parseInt(e.target.value))}
            className="border p-2 rounded w-full"
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
            checked={allowAdvancedJack}
            onChange={(e) => setAllowAdvancedJack(e.target.checked)}
          />
          <label>Allow advanced jack (remove from sequences)</label>
        </div>
        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={allowDraws}
            onChange={(e) => setAllowDraws(e.target.checked)}
          />
          <label>Allow draws (optional)</label>
        </div>
        <div>
          <label className="block font-medium">Turn time (seconds):</label>
          <input
            type="number"
            min={10}
            value={turnSeconds}
            onChange={(e) => setTurnSeconds(parseInt(e.target.value))}
            className="border p-2 rounded w-full"
          />
        </div>
        <div>
          <label className="block font-medium">Total game time (minutes):</label>
          <input
            type="number"
            min={5}
            value={totalMinutes}
            onChange={(e) => setTotalMinutes(parseInt(e.target.value))}
            className="border p-2 rounded w-full"
          />
        </div>
        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={isPublic}
            onChange={(e) => setIsPublic(e.target.checked)}
          />
          <label>Public room (no join code required)</label>
        </div>
        <div>
          <label className="block font-medium">Agent weights file (optional):</label>
          <input
            type="file"
            accept=".pt,.onnx"
            onChange={(e) => setWeightsFile(e.target.files?.[0] || null)}
            className="w-full"
          />
        </div>
        <button
          type="submit"
          disabled={creating}
          className="py-2 px-4 bg-indigo-600 text-white rounded hover:bg-indigo-700"
        >
          {creating ? 'Creating...' : 'Create Game'}
        </button>
      </form>
    </div>
  );
};

export default NewGamePage;
