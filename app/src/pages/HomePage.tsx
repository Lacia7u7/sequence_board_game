import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { signInAsGuest } from '../lib/firebase';

const HomePage: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const handleGuest = async () => {
    await signInAsGuest();
    navigate('/new');
  };
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-purple-500 to-indigo-600 text-white p-4">
      <h1 className="text-4xl font-bold mb-6 drop-shadow-md">Sequence Online</h1>
      <div className="space-y-4 w-full max-w-xs">
        <button
          className="w-full py-2 px-4 rounded bg-white text-indigo-700 font-semibold hover:bg-gray-100"
          onClick={() => navigate('/login')}
        >
          {t('login')}
        </button>
        <button
          className="w-full py-2 px-4 rounded bg-white text-indigo-700 font-semibold hover:bg-gray-100"
          onClick={() => navigate('/signup')}
        >
          {t('signup')}
        </button>
        <button
          className="w-full py-2 px-4 rounded border border-white text-white hover:bg-white hover:text-indigo-700"
          onClick={handleGuest}
        >
          {t('guest')}
        </button>
      </div>
    </div>
  );
};

export default HomePage;
