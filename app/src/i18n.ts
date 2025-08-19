import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

// English and Spanish translations
const resources = {
  en: {
    translation: {
      welcome: 'Welcome to Sequence Online',
      login: 'Login',
      signup: 'Sign Up',
      guest: 'Continue as Guest',
      newGame: 'New Game',
      profile: 'Profile',
      logout: 'Logout',
      start: 'Start',
      joinGame: 'Join Game',
      game: 'Game',
      lobby: 'Lobby',
      yourTurn: 'Your turn',
      waitTurn: 'Waiting for other players...',
      createGame: 'Create Game',
      seats: 'Seats',
      players: 'Players',
      matchCreated: 'Match Created',
      goToLobby: 'Go to Lobby',
    },
  },
  es: {
    translation: {
      welcome: 'Bienvenido a Sequence Online',
      login: 'Iniciar sesión',
      signup: 'Registrarse',
      guest: 'Continuar como invitado',
      newGame: 'Nueva partida',
      profile: 'Perfil',
      logout: 'Cerrar sesión',
      start: 'Comenzar',
      joinGame: 'Unirse a la partida',
      game: 'Juego',
      lobby: 'Sala',
      yourTurn: 'Tu turno',
      waitTurn: 'Esperando a otros jugadores...',
      createGame: 'Crear partida',
      seats: 'Asientos',
      players: 'Jugadores',
      matchCreated: 'Partida creada',
      goToLobby: 'Ir a la sala',
    },
  },
};

i18n.use(initReactI18next).init({
  resources,
  lng: 'en',
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },
});

export default i18n;
