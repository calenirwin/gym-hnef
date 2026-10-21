'use strict';
const $ = id => document.getElementById(id);
const STORAGE = 'hnef-table-v1';
const defaults = {ruleset: 'mini-v1', mode: 'attackers', policy: 'heuristic', seed: 42};
let config = {...defaults}, live = null, view = null, selected = null;
let busy = false, paused = false, flipped = false, timer = null, requestId = 0;
let focusSquare = [0, 0];
let rendered = null, animating = false, motionId = 0, retryAction = null, lastFailure = null;
let drag = null, skipNextMotion = false;
let reviewMode = false;
const motion = window.HnefMotion;
const same = (a, b) => a && b && a[0] === b[0] && a[1] === b[1];
const name = side => side === 'attackers' ? 'Attackers' : 'Defenders';
const policyName = policy => ({random: 'Random', heuristic: 'Heuristic', mcts: 'MCTS'})[policy];
const reviewing = () => reviewMode || live && view && view.ply !== live.ply;
const humanTurn = () => config.mode === 'local' || config.mode === view?.player;
const finished = () => view?.outcome || view?.truncated;
const coordinate = ([r, c]) => `${String.fromCharCode(97 + c)}${view.board.length - r}`;

function formConfig() {
  return {ruleset: $('ruleset').value, mode: $('mode').value, policy: $('policy').value, seed: Number($('seed').value)};
}
function fillForm() {
  $('legacy-ruleset').hidden = config.ruleset !== 'copenhagen-experimental-v1';
  for (const key of Object.keys(defaults)) $(key).value = config[key];
  formHelp();
}
function formHelp() {
  $('policy-learn').href = `/learn#${$('policy').value}`;
  $('policy-learn').hidden = $('mode').value === 'local';
  $('policy-help').textContent = {
    random: 'Unpredictable moves. A gentle place to start.',
    heuristic: 'A quick, tactical reference opponent.',
    mcts: 'Looks ahead · 32 simulations per move. May take a moment.'
  }[$('policy').value];
  if ($('mode').value === 'watch') $('policy-help').textContent += ' Both sides use this bot.';
  if ($('mode').value === 'local') $('policy-help').textContent = 'Share the board. You control both sides.';
  $('policy').disabled = busy || $('mode').value === 'local';
  $('seed-summary').textContent = $('seed').value;
}
function persist() {
  try {
    localStorage.setItem(STORAGE, JSON.stringify({config, actions: live.actions, replay: live.replay, flipped}));
    $('save-note').hidden = true;
    $('save-note').classList.remove('error');
  } catch {
    $('save-note').hidden = false;
    $('save-note').textContent = 'Autosave unavailable. Save a replay to keep this game.';
    $('save-note').classList.add('error');
  }
}
function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
  $('retry').hidden = !message || !retryAction;
  for (const element of document.querySelectorAll('[data-request-error]')) {
    element.textContent = message;
    element.hidden = !message;
  }
}
function announce(message) {
  $('announcement').textContent = message;
}
async function request(operation = 'state', actions = live?.actions || [], extra = {}, replace = true) {
  clearTimeout(timer);
  const id = ++requestId;
  busy = true;
  retryAction = null; lastFailure = null;
  showError('');
  render();
  try {
    const response = await fetch('/api/game', {
      signal: AbortSignal.timeout(30000),
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...config, operation, actions, ...extra})
    });
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || 'The move could not be completed.');
      error.invalid = response.status >= 400 && response.status < 500;
      throw error;
    }
    if (id !== requestId) return false;
    view = data;
    if (operation === 'import') config.ruleset = data.ruleset;
    if (replace) { live = data; persist(); }
    selected = null;
    return true;
  } catch (error) {
    if (id === requestId) {
      paused = true;
      skipNextMotion = false;
      lastFailure = error.invalid ? 'invalid' : 'connection';
      retryAction = () => request(operation, actions, extra, replace);
      showError(error.invalid ? error.message : 'The table is offline. Your saved game is safe.');
    }
    return false;
  } finally {
    if (id === requestId) { busy = false; render(); scheduleBot(); }
  }
}
function scheduleBot() {
  clearTimeout(timer);
  if (!view || busy || animating || paused || document.hidden || reviewing() || finished() || humanTurn() || document.querySelector('dialog[open]')) return;
  timer = setTimeout(() => request('ai'), 650);
}
function render() {
  document.body.classList.toggle('is-busy', busy);
  for (const element of $('setup-form').elements) element.disabled = busy;
  formHelp();
  if (!view) {
    $('status').textContent = busy ? '…' : 'Ready';
    for (const id of ['back', 'forward', 'play-pause', 'live', 'undo', 'download']) $(id).disabled = true;
    return;
  }
  const isReview = reviewing();
  $('experimental').hidden = config.ruleset === 'mini-v1';
  $('result-button').hidden = !finished() || isReview;
  const pieces = view.board.join('');
  $('attacker-count').textContent = [...pieces].filter(p => p === 'A').length;
  $('defender-count').textContent = [...pieces].filter(p => p === 'D' || p === 'K').length;
  for (const side of ['attacker', 'defender']) {
    const active = (view.outcome?.winner || view.player) === `${side}s`;
    $(`${side}-seat`).classList.toggle('active', active);
    $(`${side}-seat`).setAttribute('aria-label', `${name(`${side}s`)}${active ? ', active side' : ''}`);
  }
  $('ply').textContent = `${view.ply} / ${live.ply}`;
  $('review-position').hidden = !isReview;
  $('review-position').textContent = `${view.ply} / ${live.ply}`;
  $('review-controls').hidden = !isReview;
  $('review-start').disabled = $('review-back').disabled = busy || view.ply === 0;
  $('review-forward').disabled = busy || view.ply === live.ply;
  $('review-slider').disabled = busy || live.ply === 0;
  $('review-slider').max = live.ply;
  $('review-slider').value = view.ply;
  $('review-slider').setAttribute('aria-valuetext', `Move ${view.ply} of ${live.ply}`);
  $('live').hidden = !isReview;
  if (view.outcome) {
    $('status').textContent = view.outcome.winner ? `${name(view.outcome.winner)} win` : 'Draw';
    $('status').title = view.outcome.reason.replaceAll('_', ' ');
  } else {
    $('status').title = '';
    $('status').textContent = view.truncated ? 'Session limit · unfinished'
      : isReview ? 'Review' : busy ? '…'
      : humanTurn() ? (config.mode === 'local' ? `${name(view.player)} to move` : 'Your move')
      : paused ? 'Paused' : 'Thinking';
  }
  $('back').disabled = busy || view.ply === 0;
  $('forward').disabled = busy || (isReview ? view.ply === live.ply : finished() || humanTurn());
  $('live').disabled = busy || !isReview;
  const pauseLabel = paused ? 'Resume bots' : 'Pause bots';
  $('play-pause').setAttribute('aria-label', pauseLabel);
  $('play-pause').dataset.tip = paused ? 'Resume' : 'Pause';
  $('play-pause').innerHTML = paused
    ? '<svg viewBox="0 0 24 24"><path d="m8 5 10 7-10 7Z"/></svg>'
    : '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>';
  $('play-pause').hidden = config.mode !== 'watch' && (!paused || humanTurn());
  $('play-pause').disabled = isReview || !!finished() || config.mode === 'local';
  $('undo').disabled = busy || isReview || live.ply === 0 ||
    (['attackers', 'defenders'].includes(config.mode) && !live.moves.some(m => m.player === config.mode));
  $('download').disabled = busy;
  renderBoard();
  renderJournal();
}
function renderBoard() {
  const board = $('board'), n = view.board.length;
  const focused = board.contains(document.activeElement);
  const changed = rendered?.key !== view.key || rendered?.flipped !== flipped;
  const advancing = rendered && !drag && rendered.flipped === flipped &&
    view.ply === rendered.ply + 1 && view.replay.state_keys[rendered.ply] === rendered.key;
  if (changed) { motion.stop(); animating = false; motionId += 1; }
  const before = advancing && !skipNextMotion ? motion.capture(board) : null;
  if (changed) skipNextMotion = false;
  const oldCells = new Map([...board.querySelectorAll('.square')].map(cell => [
    `${cell.dataset.row},${cell.dataset.col}`, cell,
  ]));
  const rebuild = rendered?.size !== n;
  const order = Array.from({length: n}, (_, i) => flipped ? n - i - 1 : i);
  board.classList.toggle('large', n > 5);
  board.setAttribute('aria-busy', String(busy));
  board.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  board.style.gridTemplateRows = `repeat(${n}, 1fr)`;
  board.setAttribute('aria-rowcount', n);
  board.setAttribute('aria-colcount', n);
  const last = view.moves.at(-1);
  const destinations = selected ? view.legal.filter(m => same(m.source, selected)) : [];
  const canPlay = !busy && !reviewing() && !finished() && humanTurn();
  const fragment = document.createDocumentFragment();
  for (const r of order) {
    const rank = document.createElement('div');
    rank.className = 'board-rank'; rank.setAttribute('role', 'row');
    fragment.append(rank);
    for (const c of order) {
      const square = [r, c], piece = view.board[r][c];
      const existing = !rebuild && oldCells.get(`${r},${c}`);
      const button = existing || document.createElement('button');
      button.className = 'square';
      button.dataset.row = r; button.dataset.col = c;
      button.setAttribute('role', 'gridcell');
      button.setAttribute('aria-rowindex', order.indexOf(r) + 1);
      button.setAttribute('aria-colindex', order.indexOf(c) + 1);
      if ((r + c) % 2) button.classList.add('checker');
      if (same(last?.source, square)) button.classList.add('last', 'last-source');
      if (same(last?.destination, square)) button.classList.add('last', 'last-destination');
      if (same(selected, square)) button.classList.add('selected');
      button.setAttribute('aria-selected', String(!!same(selected, square)));
      const legal = destinations.some(m => same(m.destination, square));
      if (legal) button.classList.add('legal');
      const movable = canPlay && view.legal.some(m => same(m.source, square));
      if (movable) button.classList.add('movable');
      const escape = n === 5 ? r === 0 || c === 0 || r === n - 1 || c === n - 1
        : (r === 0 || r === n - 1) && (c === 0 || c === n - 1);
      if (escape) button.classList.add('escape');
      if (!existing && (r === Math.floor(n / 2) && c === Math.floor(n / 2)
          || escape && (r === 0 || r === n - 1) && (c === 0 || c === n - 1))) {
        const mark = document.createElement('span');
        mark.className = 'square-mark'; mark.setAttribute('aria-hidden', 'true');
        mark.innerHTML = escape
          ? '<svg viewBox="0 0 24 24"><path d="M6 18 18 6M8 6h10v10"/></svg>'
          : '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="m12 2 2.5 7.5L22 12l-7.5 2.5L12 22l-2.5-7.5L2 12l7.5-2.5Z"/><circle cx="12" cy="12" r="2"/></svg>';
        button.append(mark);
      }
      if (!existing || button.dataset.piece !== piece) {
        button.querySelector('.piece')?.remove();
        if (piece !== '.') {
          const disc = document.createElement('span');
          disc.className = `piece ${piece === 'A' ? 'attacker' : 'defender'}${piece === 'K' ? ' king' : ''}`;
          disc.setAttribute('aria-hidden', 'true');
          const art = document.createElement('img');
          art.src = `/assets/${{A: 'attacker', D: 'defender', K: 'king'}[piece]}.png`;
          art.alt = ''; art.draggable = false;
          disc.append(art); button.append(disc);
        }
        button.dataset.piece = piece;
      }
      if (view.outcome && !reviewing() && (piece === 'K' && view.outcome.winner === 'defenders'
          || same(last?.destination, square) && view.outcome.winner === 'attackers')) {
        button.classList.add('winning-square');
      }
      const label = {'.': 'empty', A: 'attacker', D: 'defender', K: 'king'}[piece];
      button.setAttribute('aria-label', `${coordinate(square)}, ${label}${legal ? ', legal destination' : ''}${escape ? ', escape square' : ''}`);
      button.tabIndex = same(focusSquare, square) ? 0 : -1;
      if (!existing) {
        button.addEventListener('click', () => selectSquare(square));
        button.addEventListener('keydown', event => boardKey(event, square));
      }
      rank.append(button);
    }
  }
  board.replaceChildren(fragment);
  rendered = {key: view.key, ply: view.ply, size: n, flipped};
  if (focused) focusCell();
  if (advancing) {
    animating = true;
    const id = motionId;
    motion.play(board, before, last).finally(() => {
      if (id === motionId) { animating = false; scheduleBot(); }
    });
    announce(`${name(last.player)}: ${coordinate(last.source)} to ${coordinate(last.destination)}${last.captured.length ? `, ${last.captured.length} captured` : ''}.`);
  }
}

function focusCell() {
  $('board').querySelector(`[data-row="${focusSquare[0]}"][data-col="${focusSquare[1]}"]`)?.focus({preventScroll: true});
}
function boardKey(event, square) {
  if (reviewing()) return;
  if (event.key === 'Escape') {selected = null; render(); return;}
  const steps = {ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1]};
  if (!steps[event.key]) return;
  event.preventDefault();
  const step = steps[event.key], direction = flipped ? -1 : 1, n = view.board.length;
  focusSquare = square.map((v, i) => Math.max(0, Math.min(n - 1, v + step[i] * direction)));
  renderBoard(); focusCell();
}
function selectSquare(square) {
  focusSquare = square;
  if (busy || animating || reviewing() || finished() || !humanTurn() || suppressClick) return;
  const move = selected && view.legal.find(m => same(m.source, selected) && same(m.destination, square));
  if (move) {request('move', live.actions, {action: move.action}); return;}
  if (selected && !same(selected, square) && !view.legal.some(m => same(m.source, square))) {
    announce('Choose a marked destination. Escape cancels the selection.');
    return;
  }
  selected = same(selected, square) ? null : view.legal.some(m => same(m.source, square)) ? square : null;
  render(); focusCell();
}
function renderJournal() {
  const container = $('move-list');
  if (!live.moves.length) {
    container.innerHTML = '<div class="empty-journal">—</div>';
    return;
  }
  const fragment = document.createDocumentFragment();
  live.moves.forEach((move, i) => {
    const button = document.createElement('button');
    button.className = `move-entry${i + 1 === view.ply ? ' current' : ''}`;
    button.disabled = busy;
    const number = document.createElement('span'); number.className = 'move-number'; number.textContent = String(i + 1).padStart(2, '0');
    const side = document.createElement('span'); side.className = 'move-side'; side.textContent = move.player === 'attackers' ? '● ATK' : '○ DEF';
    const notation = document.createElement('span'); notation.className = 'notation'; notation.textContent = `${coordinate(move.source)} → ${coordinate(move.destination)}`;
    button.append(number, side, notation);
    if (move.captured.length) {const captured = document.createElement('span'); captured.className = 'capture-count'; captured.textContent = `×${move.captured.length}`; button.append(captured);}
    button.setAttribute('aria-label', `Ply ${i + 1}, ${move.player}, ${coordinate(move.source)} to ${coordinate(move.destination)}, ${move.captured.length} captured. Review position.`);
    button.addEventListener('click', async () => { await review(i + 1); $('history-dialog').close(); });
    fragment.append(button);
  });
  const scroll = container.scrollTop;
  container.replaceChildren(fragment);
  container.scrollTop = reviewing() ? scroll : container.scrollHeight;
}
async function review(ply) {
  if (busy) return;
  reviewMode = true;
  paused = true; clearTimeout(timer);
  if (ply === live.ply) { view = live; selected = null; render(); }
  else await request('state', live.actions.slice(0, Math.max(0, ply)), {}, false);
}
let suppressClick = false;
function cancelDrag() {
  if (!drag) return;
  drag.ghost?.remove();
  if (drag.piece) drag.piece.style.visibility = '';
  if ($('board').hasPointerCapture(drag.id)) $('board').releasePointerCapture(drag.id);
  drag = null;
  $('board').querySelectorAll('.drop-target').forEach(cell => cell.classList.remove('drop-target'));
}
$('board').addEventListener('pointerdown', event => {
  if (event.button !== 0 || busy || animating || reviewing() || finished() || !humanTurn()) return;
  const cell = event.target.closest('.square.movable');
  if (!cell) return;
  const source = [Number(cell.dataset.row), Number(cell.dataset.col)];
  drag = {id: event.pointerId, source, x: event.clientX, y: event.clientY, active: false};
});
$('board').addEventListener('pointermove', event => {
  if (!drag || event.pointerId !== drag.id) return;
  if (!drag.active && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 7) return;
  if (!drag.active) {
    drag.active = true; selected = drag.source; focusSquare = drag.source;
    render();
    const piece = $('board').querySelector(`[data-row="${drag.source[0]}"][data-col="${drag.source[1]}"] .piece`);
    const rect = piece.getBoundingClientRect();
    drag.piece = piece; drag.rect = rect;
    drag.ghost = piece.cloneNode(true); drag.ghost.className = 'drag-piece';
    Object.assign(drag.ghost.style, {width: `${rect.width}px`, height: `${rect.height}px`});
    document.body.append(drag.ghost); piece.style.visibility = 'hidden';
    $('board').setPointerCapture(event.pointerId);
  }
  event.preventDefault();
  Object.assign(drag.ghost.style, {
    left: `${event.clientX - drag.rect.width / 2}px`, top: `${event.clientY - drag.rect.height / 2}px`,
  });
  const target = document.elementFromPoint(event.clientX, event.clientY)?.closest('.square.legal');
  $('board').querySelectorAll('.drop-target').forEach(cell => cell.classList.remove('drop-target'));
  target?.classList.add('drop-target');
});
$('board').addEventListener('pointerup', event => {
  if (!drag || event.pointerId !== drag.id) return;
  if (!drag.active) { drag = null; return; }
  const source = drag.source;
  const target = document.elementFromPoint(event.clientX, event.clientY)?.closest('.square');
  const destination = target ? [Number(target.dataset.row), Number(target.dataset.col)] : null;
  const move = view.legal.find(m => same(m.source, source) && same(m.destination, destination));
  suppressClick = true;
  cancelDrag();
  setTimeout(() => { suppressClick = false; }, 0);
  if (move) { skipNextMotion = true; request('move', live.actions, {action: move.action}); }
  else { announce('Piece returned. Choose a marked square.'); render(); }
});
$('board').addEventListener('pointercancel', () => { cancelDrag(); render(); });
addEventListener('blur', cancelDrag);
addEventListener('resize', cancelDrag);

$('setup-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  const previous = config;
  config = formConfig(); paused = false; selected = null; focusSquare = [0, 0];
  const ok = await request('state', []);
  if (!ok) {config = previous; retryAction = null; showError($('error').textContent); fillForm(); render();}
  else { reviewMode = false; $('setup-dialog').close(); }
});
for (const id of ['mode', 'policy', 'seed']) $(id).addEventListener('change', formHelp);
$('back').addEventListener('click', () => review(view.ply - 1));
$('forward').addEventListener('click', () => {
  if (reviewing()) review(view.ply + 1);
  else {paused = true; request('ai');}
});
$('live').addEventListener('click', () => {reviewMode = false; view = live; selected = null; paused = config.mode === 'watch'; render(); scheduleBot();});
$('review-start').addEventListener('click', () => review(0));
$('review-back').addEventListener('click', () => review(view.ply - 1));
$('review-forward').addEventListener('click', () => review(view.ply + 1));
$('review-slider').addEventListener('change', event => review(Number(event.target.value)));
document.addEventListener('keydown', event => {
  if (!reviewing() || busy || event.altKey || event.ctrlKey || event.metaKey || document.querySelector('dialog[open]') || event.target.matches('input, select, textarea')) return;
  const ply = {ArrowLeft: Math.max(0, view.ply - 1), ArrowRight: Math.min(live.ply, view.ply + 1), Home: 0, End: live.ply}[event.key];
  if (ply === undefined) return;
  event.preventDefault(); review(ply);
});
$('play-pause').addEventListener('click', () => {paused = !paused; clearTimeout(timer); render(); scheduleBot();});
$('undo').addEventListener('click', async () => {
  paused = config.mode === 'watch';
  let index = live.ply - 1;
  if (config.mode === 'attackers' || config.mode === 'defenders') {
    const lastHuman = live.moves.findLastIndex(m => m.player === config.mode);
    index = lastHuman < 0 ? 0 : lastHuman;
  }
  await request('state', live.actions.slice(0, index));
});
$('flip').addEventListener('click', () => {cancelDrag(); flipped = !flipped; render(); if (live) persist();});
$('download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(live.replay, null, 2) + '\n'], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = `hnef-${config.ruleset}-ply${live.ply}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('load-replay').addEventListener('click', () => $('replay-file').click());
$('replay-file').addEventListener('change', async event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file || busy) return;
  if (file.size > 120000) { importError('Replay files must be smaller than 120 KB.'); return; }
  try { await importReplay(await file.text()); }
  catch { importError('The replay file could not be read.'); }
});
$('paste-form').addEventListener('submit', event => {
  event.preventDefault();
  if (!busy) importReplay($('replay-text').value);
});
function importError(message) {
  $('import-error').textContent = message;
  $('import-error').hidden = false;
}
async function importReplay(text) {
  $('import-error').hidden = true;
  const previous = {...config};
  try {
    if (new Blob([text]).size > 120000) throw new Error('Replays must be smaller than 120 KB.');
    let replay;
    try { replay = JSON.parse(text); }
    catch { throw new Error('Choose a valid Hnef replay JSON file.'); }
    config = {...config, mode: 'local'};
    paused = true;
    if (!await request('import', [], {replay})) throw new Error($('error').textContent);
    reviewMode = true; fillForm();
    await review(0);
    $('replay-text').value = '';
    $('history-dialog').close();
    announce('Replay opened. Use Left and Right to explore the game.');
  } catch (error) {
    config = previous; retryAction = null; showError(''); render();
    importError(error.message);
  }
}
function openRules() {
  const mini = config.ruleset === 'mini-v1';
  $('rules-content').innerHTML = mini ? `<ol><li><strong>Move like a rook.</strong> Slide one piece horizontally or vertically through empty squares. Attackers go first.</li><li><strong>Close the trap.</strong> Sandwich an enemy between two of your pieces to capture it. Moving between enemies is safe. The throne can also support captures.</li><li><strong>Protect the king.</strong> Capture him on the throne with four attackers, or beside it with three. Elsewhere, two opposite attackers suffice.</li><li><strong>Race to freedom.</strong> The golden king wins by reaching any edge. Only he may stop on the central throne; other pieces may cross it when empty.</li><li><strong>Break the deadlock.</strong> No legal moves means a loss. Three occurrences of the same board and player to move means a draw.</li></ol>` : `<ol><li><strong>Move like a rook.</strong> Attackers move first. Slide through empty squares. Only the king may occupy the throne or corners; anyone may cross an empty throne.</li><li><strong>Capture by surrounding.</strong> Sandwich ordinary pieces. The king needs four adjacent attackers, or three beside the throne, and cannot be captured on an edge.</li><li><strong>Reach a corner.</strong> The king escapes at any of the four corners. The engine also recognizes some proven permanent exit forts.</li><li><strong>Mind the walls.</strong> Shield walls can capture groups along an edge. Encircling every defender wins for the attackers.</li><li><strong>Keep moving.</strong> No legal moves loses. Threefold repetition awards a win to the attackers.</li></ol><p><strong>Experimental rules:</strong> fort recognition is conservative and automatic dead-position draws are not supported. This is a playable preview, not a complete Copenhagen adjudicator.</p>`;
  if (config.ruleset === 'copenhagen-experimental-v1') {
    $('rules-content').innerHTML = $('rules-content').innerHTML.replace('The engine also recognizes some proven permanent exit forts.', 'This legacy profile does not recognize exit forts.').replace('fort recognition is conservative and automatic dead-position draws are not supported.', 'exit forts and automatic dead-position draws are not supported in legacy v1.');
  }
  openSheet('rules-dialog');
}
const outcomeReasons = {
  king_escaped: 'The king found a way through the siege.',
  king_captured: 'The ravens closed their circle around the king.',
  repetition: 'The same position occurred three times. Neither side wins.',
  defender_repetition: 'Threefold repetition awards the game to the attackers.',
  no_legal_moves: 'The opposing side has no legal move remaining.',
  encircled: 'The attackers cut every defender off from the edge.',
  exit_fort: 'The king reached a proven permanent exit fort.',
  adjudicated_draw: 'This game ended with an external draw judgment.',
};
$('result-button').addEventListener('click', () => {
  const outcome = view.outcome;
  $('result-title').textContent = outcome ? outcome.winner ? `${name(outcome.winner)} win` : 'A drawn game' : 'Unfinished';
  $('result-reason').textContent = outcome ? outcomeReasons[outcome.reason]
    : 'The 500-ply session limit was reached. This is not a draw.';
  $('result-art').src = outcome?.winner === 'attackers' ? '/assets/attacker.png' : '/assets/king.png';
  $('swap-sides').hidden = !['attackers', 'defenders'].includes(config.mode);
  openSheet('result-dialog');
});
async function rematch(swap = false) {
  if (busy) return;
  const previous = {...config};
  if (swap) config.mode = config.mode === 'attackers' ? 'defenders' : 'attackers';
  paused = false; selected = null; focusSquare = [0, 0];
  if (await request('state', [])) { reviewMode = false; fillForm(); $('result-dialog').close(); }
  else { config = previous; retryAction = null; showError($('error').textContent); render(); }
}
$('rematch').addEventListener('click', () => rematch());
$('swap-sides').addEventListener('click', () => rematch(true));
$('review-game').addEventListener('click', async () => { await review(0); $('result-dialog').close(); });
$('rules-button').addEventListener('click', openRules);
$('experimental-info').addEventListener('click', openRules);
function openSheet(id) {
  paused = true;
  clearTimeout(timer);
  selected = null;
  render();
  $(id).showModal();
}
$('new-game').addEventListener('click', () => { fillForm(); openSheet('setup-dialog'); });
$('history-button').addEventListener('click', () => openSheet('history-dialog'));
for (const button of document.querySelectorAll('[data-close]')) {
  button.addEventListener('click', () => $(button.dataset.close).close());
}
for (const id of ['setup-dialog', 'history-dialog', 'rules-dialog', 'result-dialog']) {
  $(id).addEventListener('close', () => {
    if (config.mode !== 'watch') paused = false;
    render(); scheduleBot();
  });
}
for (const id of ['close-rules', 'rules-done']) $(id).addEventListener('click', () => $('rules-dialog').close());
$('rules-dialog').addEventListener('click', event => {if (event.target === $('rules-dialog')) {const r = $('rules-dialog').getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $('rules-dialog').close();}});
async function init() {
  let actions = [], replay = null;
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE));
    if (stored && ['mini-v1', 'copenhagen-experimental-v1', 'copenhagen-experimental-v2'].includes(stored.config?.ruleset) && ['attackers', 'defenders', 'local', 'watch'].includes(stored.config?.mode) && ['random', 'heuristic', 'mcts'].includes(stored.config?.policy) && Number.isInteger(stored.config?.seed) && stored.config.seed >= 0 && stored.config.seed < 4294967296 && Array.isArray(stored.actions)) {
      config = stored.config; actions = stored.actions; flipped = !!stored.flipped; paused = config.mode === 'watch';
      replay = stored.replay || null;
    }
  } catch { /* A fresh game is still available when storage is unavailable. */ }
  fillForm();
  if (!await request(replay ? 'import' : 'state', actions, replay ? {replay} : {}) && (actions.length || replay) && lastFailure === 'invalid') {
    config = {...defaults}; fillForm();
    await request('state', []);
    showError('The saved game was invalid. A fresh table is ready.');
  }
  const lessonOpponent = new URLSearchParams(location.search).get('opponent');
  if (['random', 'heuristic', 'mcts'].includes(lessonOpponent)) {
    fillForm();
    $('policy').value = lessonOpponent;
    if (['local', 'watch'].includes($('mode').value)) $('mode').value = 'attackers';
    formHelp();
    openSheet('setup-dialog');
  }
}
$('retry').addEventListener('click', () => retryAction?.());
document.addEventListener('visibilitychange', () => {
  clearTimeout(timer);
  if (document.hidden) { motion.stop(); cancelDrag(); }
  else scheduleBot();
});
init();
