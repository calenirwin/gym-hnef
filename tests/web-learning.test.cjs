const test = require('node:test');
const assert = require('node:assert/strict');
const math = require('../src/hnef/web/learn-math.js');
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);

test('minimax accounts for the opponent rather than the best visible leaf', () => {
  assert.equal(math.minimax(-0.4).move, 'B');
  assert.equal(math.minimax(0.1).move, 'tie');
  assert.equal(math.minimax(0.5).move, 'A');
  near(math.minimax(0.9).a, 0.6);
});
test('alternating-player target reverses the opponent perspective', () => {
  near(math.qUpdate('continue', 0.6, 1).value, -0.57);
  near(math.qUpdate('continue', -0.6, 1).value, 0.57);
  near(math.qUpdate('continue', 0.6, 0).value, 0.2);
});
test('terminal updates never bootstrap from an opponent value', () => {
  near(math.qUpdate('win', -1, 1).value, 1);
  near(math.qUpdate('loss', 1, 1).value, -1);
  near(math.qUpdate('draw', 1, 1).value, 0);
  near(math.qUpdate('win', 1, 0.5).value, 0.6);
});
test('PPO clips improvement in opposite directions for opposite advantages', () => {
  near(math.ppo(1.5, 1).objective, 1.2);
  near(math.ppo(0.5, -1).objective, -0.8);
  near(math.ppo(1.5, -1).objective, -1.5);
  near(math.ppo(0.5, 1).objective, 0.5);
  near(math.ppo(1, 0).objective, 0);
});
test('the opponent mixture changes the preferred attacker', () => {
  math.mixture('uniform').forEach(value => near(value, 0.5));
  for (const [index, name] of ['x', 'y', 'z'].entries()) {
    const values = math.mixture(name);
    assert.equal(values.indexOf(Math.max(...values)), index);
    near(values[index], 0.71);
  }
});
