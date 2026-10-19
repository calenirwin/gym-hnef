let bridge, model, inference, rng;
let queue = Promise.resolve();

async function initialize() {
  self.postMessage({ status: 'Loading the game engine…' });
  const runtime = new URL('./runtime/', self.location.href).href;
  const { loadPyodide } = await import(runtime + 'pyodide.mjs');
  const py = await loadPyodide({ indexURL: runtime });
  const response = await fetch('./engine.json');
  if (!response.ok) throw new Error('Could not load the rules engine');
  const bundle = await response.json();
  for (const [name, source] of Object.entries(bundle.files)) {
    const path = '/engine/' + name;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
    py.FS.writeFile(path, source);
  }
  py.runPython("import sys; sys.path.insert(0, '/engine')\nfrom hnef.browser import command");
  bridge = py.globals.get('command');
  const modelResponse = await fetch('./model.json');
  if (!modelResponse.ok) throw new Error('Could not load the RL baseline');
  model = await modelResponse.json();
  inference = await import('./inference.mjs');
  rng = inference.seededRandom(42);
  return { ...JSON.parse(bridge('{"kind":"reset"}')), model: {
    update: model.update, checkpoint: model.checkpoint_sha256.slice(0, 12)
  }};
}

self.onmessage = event => {
  const { id, message } = event.data;
  queue = queue.then(async () => {
    try {
      let result;
      if (message.kind === 'init') result = await initialize();
      else if (!bridge) throw new Error('Engine is not ready');
      else if (message.kind === 'neural') {
        const features = JSON.parse(bridge('{"kind":"features"}'));
        const { logits } = inference.infer(model, features);
        result = JSON.parse(bridge(JSON.stringify({ kind: 'step', action: inference.sample(logits, rng) })));
      } else if (message.kind === 'replay') result = bridge(JSON.stringify(message));
      else {
        if (message.kind === 'reset') rng = inference.seededRandom(message.seed ?? 42);
        result = JSON.parse(bridge(JSON.stringify(message)));
      }
      self.postMessage({ id, result });
    } catch (error) {
      self.postMessage({ id, error: String(error.message || error) });
    }
  });
};
