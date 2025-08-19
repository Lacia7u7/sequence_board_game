import React, { useEffect, useState, useRef } from 'react';
import { collection, query, orderBy, onSnapshot } from 'firebase/firestore';
import { httpsCallable } from 'firebase/functions';
import { firestore, functions } from '../lib/firebase';
import { useAuth } from '../hooks/useAuth';

interface ChatProps {
  matchId: string;
  teamIndex: number | null;
}

interface ChatMessage {
  uid: string | null;
  displayName: string;
  text: string;
  createdAt: any;
  teamOnly?: boolean;
  teamIndex?: number;
}

const Chat: React.FC<ChatProps> = ({ matchId, teamIndex }) => {
  const { user } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [text, setText] = useState('');
  const [teamOnly, setTeamOnly] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const chatRef = collection(firestore, 'matches', matchId, 'chat');
    const q = query(chatRef, orderBy('createdAt'));
    const unsub = onSnapshot(q, (snap) => {
      const msgs: ChatMessage[] = [];
      snap.forEach((doc) => msgs.push(doc.data() as ChatMessage));
      setMessages(msgs);
    });
    return () => unsub();
  }, [matchId]);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  const sendMessage = async () => {
    if (!text.trim()) return;
    try {
      const post = httpsCallable(functions, 'post_message');
      await post({ matchId, text, teamOnly });
      setText('');
    } catch (err) {
      console.error(err);
      alert('Error sending message');
    }
  };
  return (
    <div className="flex flex-col h-96 border rounded shadow p-2 bg-white">
      <div className="flex-1 overflow-y-auto mb-2">
        {messages
          .filter((m) => !m.teamOnly || (teamOnly && teamIndex !== null && m.teamIndex === teamIndex) || (!teamOnly && !m.teamOnly))
          .map((m, idx) => (
            <div key={idx} className="mb-1 text-sm">
              <span className="font-semibold">
                {m.displayName || 'Anon'}:
              </span>{' '}
              <span>{m.text}</span>
            </div>
          ))}
        <div ref={bottomRef}></div>
      </div>
      <div className="flex items-center space-x-2">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="flex-1 border p-2 rounded"
          placeholder="Type a message"
        />
        {teamIndex !== null && (
          <label className="flex items-center space-x-1 text-xs">
            <input
              type="checkbox"
              checked={teamOnly}
              onChange={(e) => setTeamOnly(e.target.checked)}
            />
            <span>Team</span>
          </label>
        )}
        <button
          onClick={sendMessage}
          className="py-1 px-2 bg-indigo-600 text-white rounded text-sm"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default Chat;
