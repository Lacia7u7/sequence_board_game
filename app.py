from flask import Flask, jsonify, request
from sequence.game import Game

app = Flask(__name__, static_folder='frontend', static_url_path='')

game = Game()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/state')
def state():
    board = [[cell.token for cell in row] for row in game.board]
    player = game.current_player()
    return jsonify({
        'board': board,
        'currentPlayer': player.team,
        'hand': player.hand,
    })

@app.route('/api/move', methods=['POST'])
def move():
    data = request.get_json(force=True)
    card = data.get('card')
    target = data.get('target')
    if target is not None:
        target = tuple(target)
    success = game.play_turn(card, target)
    if success:
        game.next_player()
    player = game.current_player()
    board = [[cell.token for cell in row] for row in game.board]
    return jsonify({
        'success': success,
        'board': board,
        'currentPlayer': player.team,
        'hand': player.hand,
    })

if __name__ == '__main__':
    app.run(debug=True)
