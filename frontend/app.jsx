const {useState, useEffect} = React;

function App() {
  const [state, setState] = useState({board: [], hand: [], currentPlayer: ''});

  const fetchState = async () => {
    const res = await fetch('/api/state');
    const data = await res.json();
    setState(data);
  };

  useEffect(() => { fetchState(); }, []);

  const play = async (card, y, x) => {
    const res = await fetch('/api/move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({card: card, target: [y, x]})
    });
    const data = await res.json();
    setState(data);
  };

  return (
    <div>
      <h1>Sequence Web</h1>
      <div>Current player: {state.currentPlayer}</div>
      <div className="board">
        {state.board.map((row, y) => row.map((token, x) => (
          <div key={`${y}-${x}`} className="cell" onClick={() => {
            const card = prompt('Enter card to play');
            if (card) play(card, y, x);
          }}>
            {token}
          </div>
        )))}
      </div>
      <h3>Hand</h3>
      <ul>
        {state.hand.map(card => <li key={card}>{card}</li>)}
      </ul>
      <button onClick={fetchState}>Refresh</button>
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
