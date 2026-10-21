import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { infer, sample, seededRandom } from './inference.mjs';
const model = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const cases = JSON.parse(readFileSync(0, 'utf8'));
for (const example of cases) {
  const actual = infer(model, example.features);
  actual.logits.forEach((value, i) => {
    if (example.logits[i] === null) assert.equal(value, -Infinity);
    else assert.ok(Math.abs(value - example.logits[i]) < 2e-5, `Logit ${i} mismatch`);
  });
  assert.ok(Math.abs(actual.value - example.value) < 2e-5, 'Value mismatch');
  const rng = seededRandom(42);
  for (let n = 0; n < 100; n++) assert.ok(example.features.mask[sample(actual.logits, rng)]);
}
console.log(`PyTorch/browser parity passed: ${cases.length} positions`);
