const $ = id => document.getElementById(id);
const worker = new Worker('./worker.js', { type: 'module' });
const pending = new Map();
let sequence = 0, state, selected = null, busy = true, ready = false, humanTurns = [];
let gameSeed = 42, side = 'attackers', opponent = 'heuristic', mode = 'ai';
worker.onmessage = event => {
  if (event.data.status) { $('status').textContent = event.data.status; return; }
  const request = pending.get(event.data.id);
  if (!request) return;
  pending.delete(event.data.id);
  event.data.error ? request.reject(new Error(event.data.error)) : request.resolve(event.data.result);
};
worker.onerror = () => {
  const error = new Error('The game engine could not start. Check your connection and reload.');
  for (const request of pending.values()) request.reject(error);
  pending.clear(); fail(error);
};
function ask(message) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error('The engine took too long. Reload to try again.')); }, 120000);
    pending.set(id, { resolve: value => { clearTimeout(timeout); resolve(value); }, reject: error => { clearTimeout(timeout); reject(error); } });
    worker.postMessage({ id, message });
  });
}
function fail(error) {
  $('error').hidden = false;
  $('error-message').textContent = error.message;
  $('status').textContent = 'Game paused. Reload to try again.';
  busy = true;
  render();
}
function controls() {
  for (const id of ['mode', 'side', 'opponent', 'new-game', 'export']) $(id).disabled = !ready || busy;
  $('side').disabled ||= mode === 'local';
  $('opponent').disabled ||= mode === 'local';
  $('ai-settings').hidden = mode === 'local';
  $('undo').disabled = !ready || busy || !(mode === 'local' ? state?.ply : humanTurns.length);
}
function canPlay() { return mode === 'local' || state?.player === side; }
function render() {
  const board = $('board');
  board.replaceChildren();
  if (!state) {
    for (let i = 0; i < 25; i++) { const tile = document.createElement('div'); tile.className = 'square'; board.append(tile); }
    controls(); return;
  }
  const destinations = selected ? state.legal.filter(move => move.source.join() === selected.join()) : [];
  const names = { A: 'attacker', D: 'defender', K: 'king', '.': 'empty' };
  const pieceClasses = { A: 'attacker', D: 'defender', K: 'king' };
  for (let row = 0; row < 5; row++) for (let col = 0; col < 5; col++) {
    const piece = state.board[row][col], coord = 'abcde'[col] + (5 - row);
    const tile = document.createElement('button');
    tile.className = 'square';
    tile.setAttribute('aria-label', `${coord}, ${names[piece]}`);
    tile.dataset.square = `${row},${col}`;
    tile.setAttribute('aria-pressed', String(selected?.join() === [row, col].join()));
    tile.disabled = busy || !!state.outcome || state.ply >= 256 || !canPlay();
    if (row === 2 && col === 2) tile.classList.add('throne');
    if (selected?.join() === [row, col].join()) tile.classList.add('selected');
    if (destinations.some(move => move.destination.join() === [row, col].join())) tile.classList.add('destination');
    if (state.last_move?.destination.join() === [row, col].join()) tile.classList.add('last-move');
    const label = document.createElement('span'); label.className = 'coord'; label.textContent = coord; tile.append(label);
    if (piece !== '.') { const disc = document.createElement('span'); disc.className = `piece ${pieceClasses[piece]}`; disc.setAttribute('aria-hidden', 'true'); if (piece === 'K') disc.textContent = '♔'; tile.append(disc); }
    tile.onclick = () => select(row, col);
    board.append(tile);
  }
  $('ply').textContent = `PLY ${String(state.ply).padStart(2, '0')}`;
  if (state.outcome) {
    const winner = state.outcome.winner;
    $('turn-label').textContent = winner ? (mode === 'local' ? `${winner[0].toUpperCase() + winner.slice(1)} win` : `${winner === side ? 'You win' : 'Opponent wins'} · ${winner}`) : 'Draw · Threefold repetition';
    $('status').textContent = state.outcome.reason.replaceAll('_', ' ') + '. Start a new game or undo to play again.';
  } else if (state.ply >= 256) {
    $('turn-label').textContent = 'Demo move limit reached';
    $('status').textContent = 'The game is unfinished. Download the replay or start again.';
  } else {
    $('turn-label').textContent = mode === 'local' ? `${state.player[0].toUpperCase() + state.player.slice(1)} to move` : `${state.player === side ? 'Your turn' : 'Opponent’s turn'} · ${state.player}`;
    $('status').textContent = busy ? 'Opponent is thinking…' : selected ? 'Choose a highlighted destination.' : mode === 'local' ? `Pass the board to the ${state.player}. Select a piece to move.` : 'Select one of your pieces to see its legal moves.';
  }
  const pieces = state.board.join('');
  $('attackers-count').textContent = pieces.split('A').length - 1;
  $('defenders-count').textContent = pieces.split('D').length - 1;
  controls();
}
async function select(row, col) {
  if (busy || !state || state.outcome || !canPlay()) return;
  const move = selected && state.legal.find(item => item.source.join() === selected.join() && item.destination.join() === [row, col].join());
  if (!move) { selected = state.legal.some(item => item.source.join() === [row, col].join()) ? [row, col] : null; render(); return; }
  busy = true; selected = null; render();
  try {
    const before = state.ply;
    state = await ask({ kind: 'step', action: move.id }); humanTurns.push(before);
    await aiTurn();
  } catch (error) { fail(error); }
}
async function aiTurn() {
  if (mode === 'ai' && !state.outcome && state.ply < 256 && !canPlay()) {
    busy = true; render();
    state = await ask(opponent === 'neural' ? { kind: 'neural' } : { kind: 'reference', policy: opponent, seed: gameSeed * 1000 + state.ply });
  }
  busy = false; render();
}
async function newGame() {
  if (!ready || busy) return;
  busy = true; selected = null; humanTurns = []; gameSeed++;
  side = $('side').value; opponent = $('opponent').value; mode = $('mode').value;
  $('error').hidden = true;
  const notes = { heuristic: 'A one-move look ahead, guided by material and the king’s escape routes.', neural: 'A small actor–critic trained against random play. Its legal moves are sampled from the learned policy.', random: 'Uniformly sampled legal moves. A gentle place to learn the board.' };
  $('opponent-note').textContent = mode === 'local' ? 'Two people, one board. Attackers move first; take turns on this device.' : notes[opponent]; controls();
  try { state = await ask({ kind: 'reset', seed: gameSeed }); await aiTurn(); } catch (error) { fail(error); }
}
$('mode').onchange = newGame; $('side').onchange = newGame; $('opponent').onchange = newGame; $('new-game').onclick = newGame;
$('undo').onclick = async () => {
  if (busy || !state) return;
  const ply = mode === 'local' ? state.ply - 1 : humanTurns.at(-1);
  if (ply == null || ply < 0) return;
  busy = true; selected = null; render();
  try {
    state = await ask({ kind: 'undo', ply }); humanTurns.pop();
    busy = false; render();
  } catch (error) { fail(error); }
};
$('retry').onclick = () => location.reload();
$('export').onclick = async () => {
  try {
    const replay = await ask({ kind: 'replay' });
    const url = URL.createObjectURL(new Blob([replay], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `hnef-mini-${state.ply}-plies.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { fail(error); }
};
render();
try {
  state = await ask({ kind: 'init' });
  $('model-info').textContent = `RL checkpoint ${state.model.checkpoint} · ${state.model.update} updates.`;
  ready = true; busy = false; render();
} catch (error) { fail(error); }
