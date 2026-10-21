'use strict';
const transport = window.hnefTransport;

const lessons = [...document.querySelectorAll('.lesson')];
const links = [...document.querySelectorAll('.lesson-nav a')];
const lessonNames = lessons.map(lesson => links.find(link => link.hash === `#${lesson.id}`).textContent.trim());
const byId = id => document.getElementById(id);

function showLesson(focus = false) {
  const requested = location.hash.slice(1);
  const found = lessons.findIndex(lesson => lesson.id === requested);
  const index = found < 0 ? 0 : found;
  lessons.forEach((lesson, i) => { lesson.hidden = i !== index; });
  links.forEach(link => {
    if (link.hash === `#${lessons[index].id}`) link.setAttribute('aria-current', 'step');
    else link.removeAttribute('aria-current');
  });
  const previous = byId('previous-lesson'), next = byId('next-lesson');
  previous.hidden = index === 0;
  previous.href = transport.url(`learn/#${lessons[Math.max(0, index - 1)].id}`);
  previous.textContent = `← ${lessonNames[Math.max(0, index - 1)]}`;
  next.href = index === lessons.length - 1 ? transport.url('') : transport.url(`learn/#${lessons[index + 1].id}`);
  next.textContent = index === lessons.length - 1 ? 'Back to the table ↗' : `Next: ${lessonNames[index + 1]} →`;
  byId('lesson-position').textContent = `${index + 1} of ${lessons.length}`;
  byId('topic-select').value = lessons[index].id;
  document.title = `${lessonNames[index]} · Learn · Hnef`;
  if (focus) {
    lessons[index].querySelector('h2').focus({preventScroll: true});
    lessons[index].scrollIntoView({block: 'start'});
  }
}

for (const group of document.querySelectorAll('.topic-group')) {
  const optionGroup = document.createElement('optgroup');
  optionGroup.label = group.querySelector('h2').textContent;
  for (const link of group.querySelectorAll('a')) {
    const option = document.createElement('option');
    option.value = link.hash.slice(1); option.textContent = link.textContent;
    optionGroup.append(option);
  }
  byId('topic-select').append(optionGroup);
}
byId('topic-select').addEventListener('change', () => { location.hash = byId('topic-select').value; });
document.querySelector('.topic-picker').hidden = false;
document.body.classList.add('enhanced');
byId('lesson-pagination').hidden = false;
showLesson(Boolean(location.hash && lessons.some(lesson => `#${lesson.id}` === location.hash)));
window.addEventListener('hashchange', () => {
  // Preserve native anchors such as the skip link without resetting the lesson.
  if (location.hash === '#lesson-content') return;
  showLesson(true);
});

const counts = [0, 0, 0];
const sampleNames = ['A: capture a piece', 'B: open an escape lane', 'C: move away'];
function sample(amount) {
  let selected;
  for (let i = 0; i < amount; i++) {
    selected = Math.floor(Math.random() * counts.length);
    counts[selected]++;
  }
  ['a', 'b', 'c'].forEach((letter, i) => {
    const output = byId(`sample-${letter}`);
    output.textContent = `${counts[i]} selected`;
    output.parentElement.classList.toggle('picked', selected === i);
  });
  byId('sample-result').textContent = `Sampled: ${counts.reduce((a, b) => a + b, 0)}. Last: ${sampleNames[selected]}.`;
}
byId('sample-one').addEventListener('click', () => sample(1));
byId('sample-many').addEventListener('click', () => sample(100));
byId('sample-reset').addEventListener('click', () => {
  counts.fill(0);
  ['a', 'b', 'c'].forEach(letter => {
    const output = byId(`sample-${letter}`);
    output.textContent = '0 selected';
    output.parentElement.classList.remove('picked');
  });
  byId('sample-result').textContent = '0 samples.';
});

function evaluateFeatures() {
  const [material, progress, lanes, pressure] = ['material', 'progress', 'lanes', 'pressure'].map(id => Number(byId(id).value));
  byId('material-value').textContent = material.toFixed(2);
  byId('progress-value').textContent = progress.toFixed(2);
  byId('lanes-value').textContent = `${lanes} / 4`;
  byId('pressure-value').textContent = `${pressure} / 4`;
  const score = Math.max(-0.9, Math.min(0.9, 0.35 * material + 0.25 * progress + 0.30 * lanes / 4 - 0.10 * pressure / 4));
  byId('heuristic-score').textContent = `${score >= 0 ? '+' : ''}${score.toFixed(3)}`;
}
for (const id of ['material', 'progress', 'lanes', 'pressure']) byId(id).addEventListener('input', evaluateFeatures);
evaluateFeatures();

function walkthrough(prefix, descriptions, names) {
  let step = 0;
  const stages = [...byId(`${prefix}-cycle`).children];
  function render() {
    stages.forEach((stage, i) => {
      if (i === step) stage.setAttribute('aria-current', 'step');
      else stage.removeAttribute('aria-current');
    });
    byId(`${prefix}-explanation`).textContent = descriptions[step];
    byId(`${prefix}-next`).textContent = step === stages.length - 1 ? 'Repeat ↻' : `Next: ${names[step + 1]} →`;
  }
  byId(`${prefix}-next`).addEventListener('click', () => { step = (step + 1) % stages.length; render(); });
  render();
}
walkthrough('search', [
  'Select: follow branches using UCT. Opponent turns favour lower values for the root player.',
  'Expand: add one randomly selected untried move. Retain the resulting position’s full history.',
  'Roll out: play random moves until a result or the rollout limit. Score +1 for a win, −1 for a loss, and 0 for a draw, from the root player’s perspective. A cutoff contributes 0 as an unknown-value estimate.',
  'Back up: add a visit and the rollout result to each node on the path. The next simulation uses these updated statistics.'
], ['select', 'expand', 'roll out', 'back up']);
walkthrough('training', [
  'Play: guide search with policy predictions and evaluate leaf positions with the value network.',
  'Collect: save positions, history, search visits, and completed outcomes as training examples.',
  'Train: update the policy toward search visit distributions and the value toward game outcomes.',
  'Evaluate: compare the fixed checkpoint against baselines and earlier models, from both sides with recorded search budgets.'
], ['play', 'collect', 'train', 'evaluate']);

for (const button of document.querySelectorAll('[data-answer]')) {
  button.addEventListener('click', () => {
    for (const option of document.querySelectorAll('[data-answer]')) option.setAttribute('aria-pressed', String(option === button));
    byId('network-answer').textContent = button.dataset.answer === 'no'
      ? 'Correct. Policy weights express move preference. The value output estimates the outcome.'
      : 'The 60% is a move preference, not a win rate. The value output estimates the outcome.';
  });
}

const signed = (number, digits = 2) => `${number >= 0 ? '+' : ''}${number.toFixed(digits)}`;
function updateMinimax() {
  const reply = Number(byId('minimax-reply').value), result = HnefLearning.minimax(reply);
  byId('minimax-value').textContent = signed(reply, 1);
  byId('minimax-worst').textContent = signed(result.a, 1);
  byId('minimax-result').textContent = result.move === 'tie' ? 'Both moves have the same worst-reply score.' : `Choose ${result.move}: its worst reply is better.`;
}
byId('minimax-reply').addEventListener('input', updateMinimax);
updateMinimax();
function updateQ() {
  const ending = byId('q-ending').value, next = Number(byId('q-next').value), rate = Number(byId('q-rate').value);
  const result = HnefLearning.qUpdate(ending, next, rate);
  byId('q-next').disabled = ending !== 'continue';
  byId('q-next-value').textContent = signed(next);
  byId('q-rate-value').textContent = rate.toFixed(2);
  byId('q-result').textContent = `Target ${signed(result.target, 3)}. Updated Q ${signed(result.value, 3)}.${ending === 'continue' ? ' The sign changes because the next value belongs to the opponent.' : ' No bootstrap at a terminal state.'}`;
}
for (const id of ['q-ending', 'q-next', 'q-rate']) byId(id).addEventListener('input', updateQ);
updateQ();
function updatePPO() {
  const ratio = Number(byId('ppo-ratio').value), advantage = Number(byId('ppo-advantage').value);
  const result = HnefLearning.ppo(ratio, advantage);
  byId('ppo-ratio-value').textContent = ratio.toFixed(2);
  byId('ppo-advantage-value').textContent = signed(advantage);
  byId('ppo-result').textContent = `Unclipped ${signed(result.raw, 3)}. Objective ${signed(result.objective, 3)}. ${result.limited ? 'Clipping limits this sample’s improvement incentive.' : 'The unclipped term determines the objective.'}`;
}
for (const id of ['ppo-ratio', 'ppo-advantage']) byId(id).addEventListener('input', updatePPO);
updatePPO();
function updateMixture() {
  const values = HnefLearning.mixture(byId('pool-mixture').value);
  byId('pool-result').textContent = `Expected win rates: ${values.map((value, i) => `${'ABC'[i]} ${(value * 100).toFixed(0)}%`).join(', ')}. Opponent selection changes the comparison.`;
}
byId('pool-mixture').addEventListener('change', updateMixture);
updateMixture();

let experimentRunning = false, experimentURL = null;
function clearExperimentExport() {
  if (experimentURL) URL.revokeObjectURL(experimentURL);
  experimentURL = null;
  byId('experiment-download').removeAttribute('href');
  byId('experiment-json').value = '';
}
const moveName = (move, size) => {
  const square = ([r, c]) => `${String.fromCharCode(97 + c)}${size - r}`;
  return `${square(move.source)}–${square(move.destination)}`;
};
function rowCells(values, header = false) {
  const row = document.createElement('tr');
  values.forEach((value, i) => {
    const cell = document.createElement(header && i === 0 ? 'th' : 'td');
    if (header && i === 0) cell.scope = 'row';
    cell.textContent = value;
    row.append(cell);
  });
  return row;
}
function drawExperiment(result) {
  const position = result.position, size = position.board.length;
  const board = byId('experiment-board');
  board.replaceChildren();
  board.classList.toggle('large', size === 11);
  const squareNames = [], pieces = {A: ['attacker', 'Attacker'], D: ['defender', 'Defender'], K: ['king', 'King']};
  position.board.forEach((rank, r) => {
    const row = document.createElement('div'); row.className = 'experiment-rank';
    [...rank].forEach((piece, c) => {
      const square = document.createElement('span');
      square.className = `experiment-square${(r + c) % 2 ? ' shaded' : ''}`;
      if (pieces[piece]) {
        const [asset, label] = pieces[piece];
        const img = document.createElement('img'); img.src = transport.url(`assets/${asset}.png`); img.alt = '';
        square.append(img);
        squareNames.push(`${label} ${String.fromCharCode(97 + c)}${size - r}`);
      }
      row.append(square);
    });
    board.append(row);
  });
  board.setAttribute('aria-label', `${position.ruleset}, ${position.player} to move. ${squareNames.join('; ')}.`);
  byId('experiment-description').textContent = `${position.ruleset} · ply ${position.ply} · ${position.player} to move · seed ${result.seed} · ${result.iterations} simulations · ${result.rollout_plies} rollout plies`;
  byId('experiment-decisions').replaceChildren(...result.decisions.map(decision => rowCells([decision.policy, moveName(decision, size), signed(decision.heuristic_score, 3)], true)));
  const children = [...result.search.children].sort((a, b) => b.visits - a.visits || b.mean_value - a.mean_value || a.action - b.action);
  byId('experiment-children').replaceChildren(...children.map(child => rowCells([moveName(child, size), child.visits, signed(child.mean_value, 3)])));
  byId('experiment-search-summary').textContent = `${children.length} of ${result.legal_count} legal root moves explored. ${result.search.rollout_cutoffs} of ${result.iterations} rollouts ended at the limit. Mean utility includes neutral cutoff estimates.`;
  const report = JSON.stringify(result, null, 2);
  clearExperimentExport();
  byId('experiment-json').value = report;
  experimentURL = URL.createObjectURL(new Blob([report], {type: 'application/json'}));
  byId('experiment-download').href = experimentURL;
  byId('experiment-download').download = `hnef-analysis-${position.ruleset}-ply${position.ply}.json`;
  byId('experiment-results').hidden = false;
}
byId('experiment-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (experimentRunning) return;
  experimentRunning = true;
  byId('experiment-cancel').hidden = transport.mode !== 'worker';
  for (const field of byId('experiment-form').elements) field.disabled = true;
  byId('experiment-results').hidden = true;
  byId('experiment-error').hidden = true;
  byId('experiment-status').textContent = 'Analyzing…';
  clearExperimentExport();
  try {
    const payload = {seed: Number(byId('experiment-seed').value), iterations: Number(byId('experiment-budget').value), rollout_plies: Number(byId('experiment-rollout').value)};
    const selection = byId('experiment-position').value;
    if (selection === 'saved') {
      let stored;
      try { stored = JSON.parse(localStorage.getItem('hnef-table-v1')); }
      catch { throw new Error('The browser save is unavailable. Choose an opening.'); }
      if (!stored) throw new Error('No saved game. Choose an opening or play a game first.');
      if (stored.replay) payload.replay = stored.replay;
      else if (stored.config?.ruleset && Array.isArray(stored.actions)) { payload.ruleset = stored.config.ruleset; payload.actions = stored.actions; }
      else throw new Error('The browser save is invalid. Choose an opening.');
    } else payload.ruleset = selection;
    const response = await transport.request('analyze', payload, {timeout: 60000});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Analysis failed.');
    drawExperiment(result);
    byId('experiment-status').textContent = 'Analysis complete. Your game is unchanged.';
  } catch (error) {
    byId('experiment-status').textContent = '';
    byId('experiment-error').textContent = error.name === 'TimeoutError' || error instanceof TypeError ? 'Analysis unavailable. Check the local server and try again.' : error.message;
    byId('experiment-error').hidden = false;
  } finally {
    experimentRunning = false;
    byId('experiment-cancel').hidden = true;
    for (const field of byId('experiment-form').elements) field.disabled = false;
  }
});
byId('experiment-form').addEventListener('input', () => {
  if (experimentRunning) return;
  clearExperimentExport();
  byId('experiment-results').hidden = true;
  byId('experiment-error').hidden = true;
  byId('experiment-status').textContent = 'Settings changed. Analyze to update results.';
});

transport.onStatus = text => { if (experimentRunning && text) byId('experiment-status').textContent = text; };
byId('experiment-cancel').addEventListener('click', () => transport.cancel());

// These small, fixed illustrations are regression-checked against the Python engine.
for (const demo of document.querySelectorAll('.tactic-demo')) {
  const board = demo.querySelector('.tactic-board');
  const button = demo.querySelector('.tactic-toggle');
  const initial = Object.fromEntries(demo.dataset.pieces.split(' ').map(piece => piece.split(':')));
  const [source, destination] = demo.dataset.move.split(' ');
  const names = {A: 'attacker', D: 'defender', K: 'king'};
  const description = board.getAttribute('aria-label');
  let after = false;
  function draw() {
    const pieces = {...initial};
    if (after) {
      pieces[destination] = pieces[source]; delete pieces[source];
      for (const square of demo.dataset.captured.split(' ')) delete pieces[square];
    }
    const cells = document.createDocumentFragment();
    for (let rank = 11; rank >= 0; rank--) {
      for (const file of ['', ...'abcdefghijk']) {
        const cell = document.createElement('span');
        if (!file || !rank) {
          cell.className = 'tactic-coordinate'; cell.textContent = file || rank || '';
        } else {
          const coordinate = `${file}${rank}`;
          cell.className = `tactic-square ${(rank + file.charCodeAt(0)) % 2 ? 'shaded' : ''}`;
          if (['a1', 'a11', 'k1', 'k11', 'f6'].includes(coordinate)) cell.classList.add('restricted');
          if (coordinate === (after ? destination : source)) cell.classList.add('tactic-mark');
          if (pieces[coordinate]) {
            const img = document.createElement('img');
            img.src = transport.url(`assets/${names[pieces[coordinate]]}.png`); img.alt = '';
            cell.append(img);
          }
        }
        cells.append(cell);
      }
    }
    board.replaceChildren(cells);
    board.setAttribute('aria-label', after ? `${source} to ${destination}. Captured: ${demo.dataset.captured}.` : description);
    button.setAttribute('aria-pressed', String(after));
    button.textContent = after ? 'Reset position' : `Show ${source} → ${destination}`;
  }
  button.addEventListener('click', () => {after = !after; draw();});
  board.hidden = false; button.hidden = false; draw();
}
