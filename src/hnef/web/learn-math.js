'use strict';

// Small teaching calculations, shared with dependency-free Node tests.
const HnefLearning = Object.freeze({
  minimax(reply) {
    const a = Math.min(0.6, reply), b = 0.1;
    return {a, b, move: a === b ? 'tie' : a > b ? 'A' : 'B'};
  },
  qUpdate(ending, opponentValue, rate) {
    const rewards = {win: 1, loss: -1, draw: 0};
    const target = ending === 'continue' ? -0.95 * opponentValue : rewards[ending];
    return {target, value: 0.2 + rate * (target - 0.2)};
  },
  ppo(ratio, advantage) {
    const raw = ratio * advantage;
    const clipped = Math.max(0.8, Math.min(1.2, ratio)) * advantage;
    return {raw, objective: Math.min(raw, clipped), limited: clipped < raw};
  },
  mixture(name) {
    const weights = {uniform: [1/3, 1/3, 1/3], x: [0.8, 0.1, 0.1], y: [0.1, 0.8, 0.1], z: [0.1, 0.1, 0.8]}[name];
    return [[0.8, 0.2, 0.5], [0.5, 0.8, 0.2], [0.2, 0.5, 0.8]].map(row => row.reduce((sum, value, i) => sum + value * weights[i], 0));
  }
});
if (typeof module !== 'undefined') module.exports = HnefLearning;
