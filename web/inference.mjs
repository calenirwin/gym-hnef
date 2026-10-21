// Portable forward pass for the versioned mini actor/critic, parity-tested against PyTorch.
function linear(x, matrix, bias) {
  return matrix.map((row, i) => row.reduce((sum, weight, j) => sum + weight * x[j], bias[i]));
}
const tanh = xs => xs.map(Math.tanh);
const sigmoid = value => 1 / (1 + Math.exp(-value));

export function infer(model, features) {
  if (model.schema !== 1 || model.features !== 'mini-board-hash-history-v1' || model.ruleset !== 'mini-v1') {
    throw new Error('Unsupported model');
  }
  const w = model.weights, width = model.network.history_width;
  let hidden = Array(width).fill(0);
  for (const position of features.history) {
    const input = linear(position, w['history.weight_ih_l0'], w['history.bias_ih_l0']);
    const recurrent = linear(hidden, w['history.weight_hh_l0'], w['history.bias_hh_l0']);
    hidden = hidden.map((previous, i) => {
      const reset = sigmoid(input[i] + recurrent[i]);
      const update = sigmoid(input[i + width] + recurrent[i + width]);
      const candidate = Math.tanh(input[i + 2 * width] + reset * recurrent[i + 2 * width]);
      return (1 - update) * candidate + update * previous;
    });
  }
  const board = tanh(linear(features.board, w['board.0.weight'], w['board.0.bias']));
  const shared = tanh(linear([...board, ...hidden], w['shared.0.weight'], w['shared.0.bias']));
  const logits = linear(shared, w['actor.weight'], w['actor.bias'])
    .map((value, i) => features.mask[i] ? value : -Infinity);
  if (!logits.some(Number.isFinite)) throw new Error('No legal moves');
  return { logits, value: Math.tanh(linear(shared, w['critic.weight'], w['critic.bias'])[0]) };
}

export function sample(logits, random) {
  const max = Math.max(...logits);
  const weights = logits.map(value => Math.exp(value - max));
  let threshold = random() * weights.reduce((a, b) => a + b, 0);
  let last = -1;
  for (let i = 0; i < weights.length; i++) {
    if (!weights[i]) continue;
    last = i;
    threshold -= weights[i];
    if (threshold < 0) return i;
  }
  return last;
}

export function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let t = Math.imul(value ^ value >>> 15, 1 | value);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
