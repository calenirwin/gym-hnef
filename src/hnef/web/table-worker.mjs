// Original Python API in a dedicated worker; no Mini bridge or alternate rules.
let bridge;
let queue = Promise.resolve();
self.onmessage = ({data: {id, kind, payload}}) => {
  queue = queue.then(async () => {
    try {
      if (kind === 'init') {
        const {loadPyodide} = await import('./runtime/pyodide.mjs');
        const py = await loadPyodide({indexURL: new URL('./runtime/', self.location.href).href});
        const response = await fetch('./engine.json', {cache: 'no-cache'});
        if (!response.ok) throw new Error('Game files could not be loaded. Check your connection and retry.');
        const bundle = await response.json();
        for (const [name, source] of Object.entries(bundle.files)) {
          if (!/^hnef\/(?:[a-z_]+\/)*[a-z_]+\.py$/.test(name)) throw new Error('Invalid engine bundle.');
          const path = '/engine/' + name;
          py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
          py.FS.writeFile(path, source);
        }
        py.runPython(`import sys, json
sys.path.insert(0, '/engine')
from hnef.web import game_response
from hnef.web.analysis import analyze_response
def table_command(text):
    request = json.loads(text)
    try:
        function = {'game': game_response, 'analyze': analyze_response}[request['kind']]
        return json.dumps({'ok': True, 'status': 200, 'data': function(request['payload'])})
    except (ValueError, TypeError, KeyError) as error:
        return json.dumps({'ok': False, 'status': 400, 'data': {'error': str(error)}})
`);
        bridge = py.globals.get('table_command');
        self.postMessage({id, ok: true});
      } else {
        if (!bridge) throw new Error('Game engine is not ready.');
        self.postMessage({id, ...JSON.parse(bridge(JSON.stringify({kind, payload})))});
      }
    } catch (error) {
      self.postMessage({id, fatal: kind === 'init'
        ? 'Game engine could not start. Check your connection and retry. Your saved game is unchanged.'
        : 'Game engine stopped. Retry to restart it. Your saved game is unchanged.'});
    }
  });
};
