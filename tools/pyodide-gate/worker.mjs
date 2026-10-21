// Feasibility probe only: run the unchanged original Python response functions.
let bridge;
let queue = Promise.resolve();
self.onmessage = ({data: {id, kind, payload}}) => {
  queue = queue.then(async () => {
    const started = performance.now();
    try {
      if (kind === 'init') {
        const {loadPyodide} = await import('./runtime/pyodide.mjs');
        const py = await loadPyodide({indexURL: new URL('./runtime/', self.location.href).href});
        const bundle = await (await fetch('./engine.json')).json();
        for (const [name, source] of Object.entries(bundle)) {
          const path = '/engine/' + name;
          py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
          py.FS.writeFile(path, source);
        }
        py.runPython(`import sys, json
sys.path.insert(0, '/engine')
from hnef.web import game_response
from hnef.web.analysis import analyze_response
def gate_command(text):
    request = json.loads(text)
    try:
        fn = analyze_response if request['kind'] == 'analyze' else game_response
        return json.dumps({'result': fn(request['payload'])})
    except (ValueError, TypeError, KeyError) as error:
        return json.dumps({'error': {'type': type(error).__name__, 'message': str(error)}})
`);
        bridge = py.globals.get('gate_command');
        self.postMessage({id, result: {python: py.runPython('sys.version')}, elapsed: performance.now()-started});
      } else {
        if (!bridge) throw new Error('Engine is not ready');
        // Intentionally synchronous Python, on the worker, never on the UI thread.
        const result = JSON.parse(bridge(JSON.stringify({kind, payload})));
        self.postMessage({id, ...result, elapsed: performance.now()-started});
      }
    } catch (error) {
      self.postMessage({id, failure: String(error.stack || error), elapsed: performance.now()-started});
    }
  });
};
