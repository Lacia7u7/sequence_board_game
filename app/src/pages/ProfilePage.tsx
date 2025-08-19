import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { doc, onSnapshot, collection, query, where, getDocs } from 'firebase/firestore';
import { firestore } from '../lib/firebase';
import { useAuth } from '../hooks/useAuth';
import { httpsCallable } from 'firebase/functions';
import { functions } from '../lib/firebase';

const ProfilePage: React.FC = () => {
  const { uid } = useParams<{ uid: string }>();
  const { user } = useAuth();
  const [profile, setProfile] = useState<any>(null);
  const [friends, setFriends] = useState<any[]>([]);
  const [sentRequests, setSentRequests] = useState<any[]>([]);
  const [receivedRequests, setReceivedRequests] = useState<any[]>([]);
  const [requestUid, setRequestUid] = useState('');
  useEffect(() => {
    if (!uid) return;
    const userRef = doc(firestore, 'users', uid);
    const unsub = onSnapshot(userRef, (snap) => {
      setProfile(snap.data());
    });
    return () => unsub();
  }, [uid]);
  useEffect(() => {
    // Load friend list
    if (!uid) return;
    const fetchFriends = async () => {
      const userDoc = await getDocs(collection(firestore, 'users'));
      const allUsers: Record<string, any> = {};
      userDoc.forEach((d) => (allUsers[d.id] = d.data()));
      if (profile?.friends) {
        const list = profile.friends.map((fid: string) => ({ uid: fid, name: allUsers[fid]?.displayName || fid }));
        setFriends(list);
      }
    };
    fetchFriends();
  }, [profile, uid]);
  useEffect(() => {
    if (!user) return;
    // Queries for pending friend requests
    const sentQ = query(collection(firestore, 'friendRequests'), where('fromUid', '==', user.uid), where('status', '==', 'pending'));
    const recQ = query(collection(firestore, 'friendRequests'), where('toUid', '==', user.uid), where('status', '==', 'pending'));
    const unsubSent = onSnapshot(sentQ, (snap) => {
      const reqs: any[] = [];
      snap.forEach((doc) => reqs.push({ id: doc.id, ...doc.data() }));
      setSentRequests(reqs);
    });
    const unsubRec = onSnapshot(recQ, (snap) => {
      const reqs: any[] = [];
      snap.forEach((doc) => reqs.push({ id: doc.id, ...doc.data() }));
      setReceivedRequests(reqs);
    });
    return () => {
      unsubSent();
      unsubRec();
    };
  }, [user]);
  const sendRequest = async () => {
    try {
      const send = httpsCallable(functions, 'send_friend_request');
      await send({ toUid: requestUid });
      setRequestUid('');
    } catch (err) {
      console.error(err);
      alert('Error sending request');
    }
  };
  const respondRequest = async (reqId: string, action: 'accept' | 'decline') => {
    try {
      const respond = httpsCallable(functions, 'respond_friend_request');
      await respond({ requestId: reqId, action });
    } catch (err) {
      console.error(err);
      alert('Error responding to request');
    }
  };
  if (!profile) return <div className="p-4">Loading...</div>;
  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Profile</h2>
      <h3 className="text-xl font-semibold mb-2">User: {profile.displayName || uid}</h3>
      <div className="mb-4">
        <h4 className="font-semibold">Stats:</h4>
        <pre className="bg-gray-100 p-2 rounded">
{JSON.stringify(profile.stats || {}, null, 2)}
        </pre>
      </div>
      <div className="mb-4">
        <h4 className="font-semibold">Friends:</h4>
        <ul className="list-disc list-inside">
          {friends.map((f) => (
            <li key={f.uid}>{f.name}</li>
          ))}
        </ul>
      </div>
      {user && user.uid === uid && (
        <div className="mb-4">
          <h4 className="font-semibold">Send Friend Request</h4>
          <input
            type="text"
            value={requestUid}
            onChange={(e) => setRequestUid(e.target.value)}
            placeholder="User UID"
            className="border p-2 rounded w-full mb-2"
          />
          <button
            onClick={sendRequest}
            className="py-2 px-4 bg-indigo-600 text-white rounded"
          >
            Send Request
          </button>
        </div>
      )}
      {user && user.uid === uid && (
        <div className="mb-4">
          <h4 className="font-semibold">Pending Friend Requests</h4>
          <ul className="space-y-2">
            {receivedRequests.map((req) => (
              <li key={req.id} className="border p-2 rounded">
                <span>{req.fromUid}</span>
                <div className="mt-1 space-x-2">
                  <button
                    onClick={() => respondRequest(req.id, 'accept')}
                    className="py-1 px-2 bg-green-600 text-white rounded text-xs"
                  >
                    Accept
                  </button>
                  <button
                    onClick={() => respondRequest(req.id, 'decline')}
                    className="py-1 px-2 bg-red-600 text-white rounded text-xs"
                  >
                    Decline
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ProfilePage;
